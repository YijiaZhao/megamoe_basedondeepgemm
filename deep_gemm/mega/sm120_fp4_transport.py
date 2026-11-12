# SM120 FP4 MegaMoE: per-pair transport table (P2P / NIC) and its setup probe.
#
# Ported from FlashInfer's Cake SM120 MegaMoE (flashinfer/moe_ep/backends/mega/kernel/sm120/mxfp4_mxfp8_bf16_cake/
# cake_transport.py, Apache-2.0): the same table kinds, candidates, validation, selection rule and probe; only the
# probe driver is adapted to this kernel's extension (`probe_round` in sm120_fp4_mega_moe.cu, the window's pool and
# GIN staging regions) and the probe traffic to this kernel's token row and partial row.
#
# The kernel serves every ordered pair of EP ranks with one of three transports: SELF (0, the diagonal), LSA (1,
# load/store over a peer mapping of the NCCL window, PCIe P2P here) or GIN (2, NCCL GPU-initiated puts through the
# NIC). The table is symmetric, row-major, world * world cells; a rank uses its row.
from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRANSPORT_SELF = 0
TRANSPORT_LSA = 1
TRANSPORT_GIN = 2

_NEAR_TIE = 0.01
SELECTION_MARGIN = 0.03
SELECTION_SPREAD_LIMIT = 1.5
_PROBE_ROUNDS = 3
_ISOLATED_BYTES = 16 << 20
_PROBE_CAP_BYTES = 32 << 20
_GIN_CHUNK_QUANTUM = 64 << 10


@dataclass(frozen=True, slots=True)
class TransportTable:
    world: int
    cells: tuple[int, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if self.world < 1 or len(self.cells) != self.world * self.world:
            raise ValueError("TransportTable cells must have world*world entries")

    def get(self, src: int, dst: int) -> int:
        return self.cells[src * self.world + dst]

    def row(self, rank: int) -> tuple[int, ...]:
        return self.cells[rank * self.world : (rank + 1) * self.world]

    def count(self, kind: int) -> int:
        return sum(1 for c in self.cells if c == kind)

    def max_row_count(self, kind: int) -> int:
        """Largest number of off-diagonal ``kind`` cells in one row."""

        return max(
            sum(1 for dst, c in enumerate(self.row(src)) if dst != src and c == kind)
            for src in range(self.world)
        )


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def table_from_pairs(
    world: int, chooser: Callable[[int, int], int], name: str = ""
) -> TransportTable:
    cells = [TRANSPORT_SELF] * (world * world)
    for a in range(world):
        for b in range(a + 1, world):
            kind = int(chooser(a, b))
            if kind not in (TRANSPORT_LSA, TRANSPORT_GIN):
                raise ValueError(f"chooser returned {kind!r} for pair {(a, b)}")
            cells[a * world + b] = kind
            cells[b * world + a] = kind
    return TransportTable(world, tuple(cells), name)


def uniform_table(world: int, kind: int) -> TransportTable:
    name = "all_lsa" if kind == TRANSPORT_LSA else "all_gin"
    return table_from_pairs(world, lambda a, b: kind, name)


def normalize_pair_ok(
    world: int, lsa_reachable_or_pairs: Any
) -> dict[tuple[int, int], bool]:
    """Normalize LSA reachability into a symmetric per-pair map.

    Accepts ``{(a, b): bool}`` (either order), a full ``world x world`` matrix,
    or a per-rank vector (expanded as ``reach[a] and reach[b]``).  Reachability is
    a property of a pair: on a multi-host deployment the per-rank vector would mark
    every rank unreachable and forbid LSA inside each host.
    """
    if isinstance(lsa_reachable_or_pairs, Mapping):
        pairs: dict[tuple[int, int], bool] = {}
        for key, value in lsa_reachable_or_pairs.items():
            a, b = (int(key[0]), int(key[1]))
            if a == b:
                continue
            if not (0 <= a < world and 0 <= b < world):
                raise ValueError(f"pair {key!r} is outside world {world}")
            pairs[_pair_key(a, b)] = bool(value)
        missing = [
            (a, b)
            for a in range(world)
            for b in range(a + 1, world)
            if (a, b) not in pairs
        ]
        if missing:
            raise ValueError(
                f"pair reachability map is missing {len(missing)} pair(s), e.g. {missing[0]}"
            )
        return pairs
    rows = list(lsa_reachable_or_pairs)
    if len(rows) != world:
        raise ValueError("lsa_reachable length must equal world")
    if rows and isinstance(rows[0], (list, tuple)):
        matrix = [list(row) for row in rows]
        for row in matrix:
            if len(row) != world:
                raise ValueError("lsa_reachable matrix rows must have world entries")
        pairs = {}
        for a in range(world):
            for b in range(a + 1, world):
                if bool(matrix[a][b]) != bool(matrix[b][a]):
                    raise ValueError(
                        f"pair reachability matrix is not symmetric at ({a},{b})"
                    )
                pairs[(a, b)] = bool(matrix[a][b])
        return pairs
    reach = [bool(v) for v in rows]
    return {
        (a, b): (reach[a] and reach[b])
        for a in range(world)
        for b in range(a + 1, world)
    }


def validate_table(
    table: TransportTable, lsa_reachable: Any, gin_available: bool
) -> None:
    world = table.world
    pair_ok = normalize_pair_ok(world, lsa_reachable)
    for a in range(world):
        for b in range(world):
            cell = table.get(a, b)
            if a == b:
                if cell != TRANSPORT_SELF:
                    raise ValueError(f"diagonal cell ({a},{a}) must be SELF")
                continue
            if cell == TRANSPORT_SELF:
                raise ValueError(f"off-diagonal cell ({a},{b}) cannot be SELF")
            if cell != table.get(b, a):
                raise ValueError(f"table is not symmetric at ({a},{b})")
            if cell == TRANSPORT_LSA and not pair_ok[_pair_key(a, b)]:
                raise ValueError(
                    f"LSA selected for pair ({a},{b}) but the pair is not load/store reachable"
                )
            if cell == TRANSPORT_GIN and not gin_available:
                raise ValueError(
                    f"GIN selected for pair ({a},{b}) but GIN is unavailable "
                    "(no GIN transport, or GIN disabled, e.g. NCCL_GIN_ENABLE=0)"
                )
            if cell not in (TRANSPORT_LSA, TRANSPORT_GIN):
                raise ValueError(f"unknown transport {cell!r} at ({a},{b})")


def lsa_where_reachable(world: int, lsa_reachable: Any) -> TransportTable:
    pair_ok = normalize_pair_ok(world, lsa_reachable)
    return table_from_pairs(
        world,
        lambda a, b: TRANSPORT_LSA if pair_ok[_pair_key(a, b)] else TRANSPORT_GIN,
        "lsa_where_reachable",
    )


def build_candidates(
    world: int,
    lsa_reachable: Any,
    gin_available: bool,
    lsa_bw: Mapping[tuple[int, int], float],
    gin_bw: Mapping[tuple[int, int], float],
) -> list[TransportTable]:
    pair_ok = normalize_pair_ok(world, lsa_reachable)

    def reach(a: int, b: int) -> bool:
        return pair_ok[_pair_key(a, b)]

    def faster_lsa(a: int, b: int) -> bool:
        key = _pair_key(a, b)
        return reach(a, b) and lsa_bw.get(key, 0.0) >= gin_bw.get(key, 0.0)

    raw = [
        uniform_table(world, TRANSPORT_GIN),
        uniform_table(world, TRANSPORT_LSA),
        lsa_where_reachable(world, pair_ok),
        table_from_pairs(
            world,
            lambda a, b: TRANSPORT_LSA if faster_lsa(a, b) else TRANSPORT_GIN,
            "lsa_where_faster",
        ),
    ]
    out: list[TransportTable] = []
    seen: set[tuple[int, ...]] = set()
    for table in raw:
        try:
            validate_table(table, pair_ok, gin_available)
        except ValueError:
            continue
        if table.cells in seen:
            continue
        seen.add(table.cells)
        out.append(table)
    if not out:
        raise ValueError(
            "no feasible transport table: some peer is neither LSA reachable nor GIN reachable"
        )
    return out


def _is_uniform(table: TransportTable) -> bool:
    return table.count(TRANSPORT_LSA) == 0 or table.count(TRANSPORT_GIN) == 0


def _spread(rounds: Sequence[float]) -> float:
    if not rounds:
        return 1.0
    lo, hi = min(rounds), max(rounds)
    if lo <= 0.0:
        return 1.0 if hi <= 0.0 else float("inf")
    return hi / lo


@dataclass(frozen=True, slots=True)
class SelectionResult:
    chosen: TransportTable
    best_uniform: TransportTable
    eligible_nonuniform: tuple[TransportTable, ...]
    candidates: tuple[dict[str, Any], ...]
    margin: float
    spread_limit: float


def evaluate_selection(
    scored: Sequence[tuple[TransportTable, float, Sequence[float]]],
    *,
    margin: float = SELECTION_MARGIN,
    spread_limit: float = SELECTION_SPREAD_LIMIT,
) -> SelectionResult:
    """Selection-stability rule: a mixed table must clearly and reproducibly win.

    ``scored`` holds per candidate the aggregate score (min over rounds of max over
    ranks) and the per-round max-over-ranks times.

    1. The best uniform table is picked among ``all_gin``/``all_lsa``.  A uniform
       candidate whose rounds spread beyond ``spread_limit`` is scored by its median
       round.  ``all_gin`` is preferred unless ``all_lsa`` beats it by ``margin``: only
       an LSA-free table lets the kernel drop its copier CTAs.  A near-tie between
       LSA-bearing uniform tables keeps the one with more LSA cells.
    2. A mixed table is eligible only if its rounds spread at most ``spread_limit``
       and its score beats the *minimum* uniform score by ``margin``.
    3. The lowest-scoring eligible mixed table wins; otherwise the best uniform one.
    """
    if not scored:
        raise ValueError("selection requires at least one scored candidate")
    candidate_meta: list[dict[str, Any]] = []
    uniform_entries: list[tuple[TransportTable, float]] = []
    nonuniform_entries: list[tuple[TransportTable, float, float]] = []
    for table, raw_score, raw_rounds in scored:
        score = float(raw_score)
        rounds = [float(r) for r in raw_rounds]
        sp = _spread(rounds)
        uniform = _is_uniform(table)
        if uniform and sp > spread_limit:
            ordered = sorted(rounds)
            mid = len(ordered) // 2
            effective = (
                ordered[mid]
                if len(ordered) % 2 == 1
                else (ordered[mid - 1] + ordered[mid]) / 2.0
            )
        else:
            effective = score
        candidate_meta.append(
            {
                "name": table.name,
                "score": effective if uniform else score,
                "raw_score": score,
                "rounds_ms": rounds,
                "spread": sp,
                "uniform": uniform,
            }
        )
        if uniform:
            uniform_entries.append((table, effective))
        else:
            nonuniform_entries.append((table, score, sp))
    if not uniform_entries:
        raise ValueError("selection requires at least one uniform candidate")
    best_uniform_score = min(score for _, score in uniform_entries)

    def _pick(
        entries: list[tuple[TransportTable, float]],
    ) -> tuple[TransportTable, float]:
        lowest = min(score for _, score in entries)
        near = [item for item in entries if item[1] <= lowest * (1.0 + _NEAR_TIE)]
        near.sort(key=lambda item: (-item[0].count(TRANSPORT_LSA), item[1]))
        return near[0]

    gin_uniform = [
        item for item in uniform_entries if item[0].count(TRANSPORT_LSA) == 0
    ]
    lsa_uniform = [
        item for item in uniform_entries if item[0].count(TRANSPORT_LSA) != 0
    ]
    if gin_uniform and lsa_uniform:
        best_gin = _pick(gin_uniform)
        best_lsa = _pick(lsa_uniform)
        best_uniform_table = (
            best_lsa[0] if best_lsa[1] <= best_gin[1] * (1.0 - margin) else best_gin[0]
        )
    else:
        best_uniform_table = _pick(uniform_entries)[0]
    eligible = [
        (table, score)
        for table, score, sp in nonuniform_entries
        if sp <= spread_limit and score <= best_uniform_score * (1.0 - margin)
    ]
    eligible.sort(key=lambda item: item[1])
    chosen = eligible[0][0] if eligible else best_uniform_table
    return SelectionResult(
        chosen=chosen,
        best_uniform=best_uniform_table,
        eligible_nonuniform=tuple(table for table, _ in eligible),
        candidates=tuple(candidate_meta),
        margin=margin,
        spread_limit=spread_limit,
    )


def derive_lsa_pair_matrix(world: int, gathered_reach: list[Any]) -> list[list[bool]]:
    """Fold every rank's local reachability vector into one symmetric pair matrix
    (a pair is usable only if both ends can map each other); identical on every rank."""
    if len(gathered_reach) != world:
        raise RuntimeError(
            f"expected {world} gathered reachability vectors, got {len(gathered_reach)}"
        )
    for r in range(world):
        if gathered_reach[r] is None or len(gathered_reach[r]) != world:
            raise RuntimeError(
                f"rank {r} reported a reachability vector of the wrong length"
            )
        if not gathered_reach[r][r]:
            raise RuntimeError(f"rank {r} does not report itself load/store reachable")
    return [
        [
            True if a == b else bool(gathered_reach[a][b] and gathered_reach[b][a])
            for b in range(world)
        ]
        for a in range(world)
    ]


def gather_lsa_pair_matrix(local_reach: Sequence[bool], group: Any) -> list[list[bool]]:
    """Collective: all ranks' reachability vectors -> the symmetric pair matrix."""
    import torch.distributed as dist

    world = dist.get_world_size(group)
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, [bool(v) for v in local_reach], group=group)
    return derive_lsa_pair_matrix(world, gathered)


def scaled_bytes_per_dest(
    *,
    world: int,
    codec_row_bytes: int,
    rows: int,
    top_k: int,
    record_bytes: int,
    cap_bytes: int = _PROBE_CAP_BYTES,
) -> int:
    """Per-destination probe bytes: one full-capacity call under uniform routing.

    ``rows * top_k / world`` routes per destination, each a dispatch record out plus an
    encoded result row back, capped at ``cap_bytes`` and rounded down to 16 bytes.
    """

    routes_per_dest = rows * top_k // world  # uniform routing
    full = routes_per_dest * (record_bytes + codec_row_bytes)
    n = min(full, cap_bytes)
    return n - (n % 16)


def gin_chunk_bytes(*, record_bytes: int, chunk_min_records: int) -> int:
    """Probe GIN put size: the kernel's minimum dispatch chunk, rounded up to 64 KiB."""

    q = _GIN_CHUNK_QUANTUM
    return (record_bytes * chunk_min_records + q - 1) // q * q


def _accumulate_targets(
    targets: list[int], row: Sequence[int], rank: int, delta: int
) -> list[int]:
    out = list(targets)
    for src in range(len(row)):
        if src != rank and row[src] == TRANSPORT_GIN:
            out[src] += delta
    return out


def probe_traffic(*, world: int, rows: int, top_k: int, hidden: int, token_row_bytes: int) -> dict[str, int]:
    """Probe traffic of this kernel: per destination one full-capacity call under uniform routing (a token row out,
    a bf16 partial row back per route); GIN puts of 64 KiB (the kernel puts one token row or one tile's partials)."""
    return {
        "bytes_per_dest": scaled_bytes_per_dest(world=world, codec_row_bytes=2 * hidden, rows=rows, top_k=top_k,
                                                record_bytes=token_row_bytes),
        "gin_chunk_bytes": gin_chunk_bytes(record_bytes=token_row_bytes, chunk_min_records=1),
    }


class _Probe:
    """One rank's probe state over the buffer's NCCL window (see the extension's probe_round)."""

    # comm: the device ncclDevComm *
    def __init__(self, ext: Any, *, group: Any, rank: int, world: int, comm: int, win: int, base: int,
                 peers: Sequence[int], lsa_reachable: Sequence[bool], gin_available: bool, gin_chunk_bytes: int):
        import torch
        import torch.distributed as dist

        self.torch, self.dist, self.ext, self.group = torch, dist, ext, group
        self.rank, self.world, self.comm, self.win = rank, world, comm, win
        pool_off, pool_bytes, src_off, src_bytes, sig_off = (int(v) for v in ext.probe_regions())
        self.src_off, self.dst_off0 = src_off, pool_off
        self.src_ptr = base + src_off
        self.src_bytes = src_bytes
        self.dst_ptr_peers = [p + pool_off if p else 0 for p in peers]
        self.peer_sig_ptrs = [p + sig_off if p else 0 for p in peers]
        self.my_sig_ptr = base + sig_off
        self.sig_bytes = 2 * int(ext.max_world) * int(ext.probe_signal_stride) * 4
        self.signal_base = int(ext.probe_signal_base)
        if gin_chunk_bytes <= 0 or gin_chunk_bytes % 16:
            raise ValueError(f"gin_chunk_bytes must be a positive multiple of 16, got {gin_chunk_bytes}")
        self.gin_chunk_bytes = gin_chunk_bytes
        self.slot_bytes = pool_bytes // world // 16 * 16
        self.epoch = 0
        self.lsa_pair_matrix = gather_lsa_pair_matrix(lsa_reachable, group)
        gathered_gin: list[Any] = [None] * world
        dist.all_gather_object(gathered_gin, bool(gin_available), group=group)
        if any(bool(v) != bool(gin_available) for v in gathered_gin):
            raise RuntimeError(f"ranks disagree about GIN availability: {[bool(v) for v in gathered_gin]}")
        self.gin_available = bool(gin_available)
        self.pair_lsa_ok = {(a, b): bool(self.lsa_pair_matrix[a][b])
                            for a in range(world) for b in range(world) if a != b}
        self.gin_targets = [0] * world
        self.barrier_targets = [0] * world
        if self.gin_available:
            self.gin_targets = [int(v) for v in ext.probe_read_gin_signals(comm, self.signal_base, world)]
            self.barrier_targets = [int(v) for v in ext.probe_read_gin_signals(comm, self.signal_base + int(ext.max_world),
                                                                                world)]
        cuda = dict(dtype=torch.int64, device='cuda')
        self._targets = torch.zeros(world, **cuda)
        self._barrier = torch.zeros(world, **cuda)
        self._scratch = torch.zeros(3, **cuda)
        self.reset_lsa_signals()
        self.last_host_ms = 0.0
        self.last_bytes = 0

    def round(self, table_row: Sequence[int], bytes_per_dest: int) -> tuple[float, int]:
        torch = self.torch
        self.epoch += 1
        effective = min(bytes_per_dest, self.slot_bytes, self.src_bytes) // 16 * 16
        chunks = (effective + self.gin_chunk_bytes - 1) // self.gin_chunk_bytes
        self.gin_targets = _accumulate_targets(self.gin_targets, table_row, self.rank, chunks)
        self.barrier_targets = _accumulate_targets(self.barrier_targets, table_row, self.rank, 1)
        self._targets.copy_(torch.tensor(self.gin_targets, dtype=torch.int64))
        self._barrier.copy_(torch.tensor(self.barrier_targets, dtype=torch.int64))
        torch.cuda.synchronize()
        self.dist.barrier(group=self.group)
        ms, host_ms = self.ext.probe_round(self.comm, self.win, self.src_ptr, self.src_off, self.dst_off0,
                                           self.dst_ptr_peers, self.peer_sig_ptrs, self.my_sig_ptr,
                                           [int(c) for c in table_row], self.rank, self.world, effective,
                                           self.slot_bytes, self.epoch, self.gin_chunk_bytes, self._targets,
                                           self._barrier, self._scratch)
        self.dist.barrier(group=self.group)
        self.last_host_ms = float(host_ms)
        self.last_bytes = effective
        return float(ms), effective

    def gather(self, value: float) -> list[float]:
        out: list[Any] = [None] * self.world
        self.dist.all_gather_object(out, value, group=self.group)
        return [float(v) for v in out]

    def reset_lsa_signals(self) -> None:
        self.torch.cuda.synchronize()
        self.torch.as_tensor(_DeviceBytes(self.my_sig_ptr, self.sig_bytes), device='cuda').zero_()
        self.torch.cuda.synchronize()
        self.dist.barrier(group=self.group)


class _DeviceBytes:
    def __init__(self, pointer: int, num_bytes: int):
        self.__cuda_array_interface__ = {'shape': (num_bytes,), 'typestr': '|u1', 'data': (pointer, False),
                                         'strides': None, 'version': 3}


def _pair_row(world: int, rank: int, a: int, b: int, kind: int) -> list[int]:
    row = [TRANSPORT_SELF] * world
    if rank == a:
        row[b] = kind
    elif rank == b:
        row[a] = kind
    return row


def _measure_isolated(probe: _Probe) -> dict[str, dict[str, float]]:
    # One discarded warm-up round per available transport (first-touch costs).
    if any(probe.pair_lsa_ok.values()):
        row = [
            TRANSPORT_LSA
            if p != probe.rank and probe.pair_lsa_ok.get((probe.rank, p), False)
            else TRANSPORT_SELF
            for p in range(probe.world)
        ]
        probe.round(row, _ISOLATED_BYTES)
    if probe.gin_available:
        row = [
            TRANSPORT_GIN if p != probe.rank else TRANSPORT_SELF
            for p in range(probe.world)
        ]
        probe.round(row, _ISOLATED_BYTES)
    out: dict[str, dict[str, float]] = {}
    for a in range(probe.world):
        for b in range(a + 1, probe.world):
            entry: dict[str, float] = {}
            if probe.pair_lsa_ok[(a, b)]:
                ms, eff = probe.round(
                    _pair_row(probe.world, probe.rank, a, b, TRANSPORT_LSA),
                    _ISOLATED_BYTES,
                )
                entry["lsa_gbps"] = eff / (max(probe.gather(ms)) * 1e6)
            if probe.gin_available:
                ms, eff = probe.round(
                    _pair_row(probe.world, probe.rank, a, b, TRANSPORT_GIN),
                    _ISOLATED_BYTES,
                )
                entry["gin_gbps"] = eff / (max(probe.gather(ms)) * 1e6)
            out[f"{a}-{b}"] = entry
    return out


def _score_candidates(
    probe: _Probe, candidates: list[TransportTable], bytes_per_dest: int
) -> list[tuple[TransportTable, float, list[float]]]:
    scored: list[tuple[TransportTable, float, list[float]]] = []
    for table in candidates:
        rounds: list[list[float]] = []
        for _ in range(_PROBE_ROUNDS):
            ms, _ = probe.round(list(table.row(probe.rank)), bytes_per_dest)
            rounds.append(probe.gather(ms))
        round_max = [max(r) for r in rounds]
        scored.append((table, min(round_max), round_max))
    return scored


def select_transport_table(ext: Any, *, group: Any, rank: int, world: int, comm: int, win: int, base: int,
                           peers: Sequence[int], lsa_reachable: Sequence[bool], gin_available: bool,
                           traffic: dict[str, int], report_path: str | None = None
                           ) -> tuple[TransportTable, dict[str, Any]]:
    """Collective setup probe: score the candidate tables and choose one (rank 0 decides). `traffic` is
    :func:`probe_traffic` of the buffer."""
    probe = _Probe(ext, group=group, rank=rank, world=world, comm=comm, win=win, base=base, peers=peers,
                   lsa_reachable=lsa_reachable, gin_available=gin_available,
                   gin_chunk_bytes=int(traffic["gin_chunk_bytes"]))
    t0 = time.time()
    isolated = _measure_isolated(probe)
    lsa_bw: dict[tuple[int, int], float] = {}
    gin_bw: dict[tuple[int, int], float] = {}
    for key, entry in isolated.items():
        a, b = (int(x) for x in key.split("-"))
        lsa_bw[(a, b)] = entry.get("lsa_gbps", 0.0)
        gin_bw[(a, b)] = entry.get("gin_gbps", 0.0)
    candidates = build_candidates(probe.world, probe.pair_lsa_ok, probe.gin_available, lsa_bw, gin_bw)
    scored = _score_candidates(probe, candidates, int(traffic["bytes_per_dest"]))
    selection = evaluate_selection(scored)
    probe.reset_lsa_signals()
    payload: list[Any] = [(selection.chosen.name, list(selection.chosen.cells)) if probe.rank == 0 else None]
    probe.dist.broadcast_object_list(payload, src=probe.dist.get_global_rank(group, 0), group=group)
    chosen_name, chosen_cells = payload[0]
    table = TransportTable(probe.world, tuple(int(c) for c in chosen_cells), chosen_name)
    validate_table(table, probe.pair_lsa_ok, probe.gin_available)
    report = {
        "host": socket.gethostname(),
        "rank": probe.rank,
        "world": probe.world,
        "lsa_pair_matrix": probe.lsa_pair_matrix,
        "gin_available": probe.gin_available,
        "isolated": isolated,
        "bytes_per_dest": probe.last_bytes,
        "gin_chunk_bytes": probe.gin_chunk_bytes,
        "candidates": [
            {"name": t.name, "cells": list(t.cells), "ms": ms, "rounds_ms": rounds, "spread": meta["spread"]}
            for (t, ms, rounds), meta in zip(scored, selection.candidates, strict=True)
        ],
        "selection_rule": {"margin": SELECTION_MARGIN, "spread_limit": SELECTION_SPREAD_LIMIT},
        "best_uniform": selection.best_uniform.name,
        "eligible_nonuniform": [t.name for t in selection.eligible_nonuniform],
        "chosen": {"name": table.name, "cells": list(table.cells)},
        "elapsed_s": time.time() - t0,
        "timing": "device_globaltimer_after_rendezvous",
        "host_ms_last": probe.last_host_ms,
    }
    if report_path and probe.rank == 0:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return table, report


# ----------------------------------------------------------------------------------------------------------------------
# Tuned tables (Cake's "tuned": the table recorded for this deployment, all_gin when none is)
# ----------------------------------------------------------------------------------------------------------------------

def tuned_table_path() -> str:
    """Env DG_SM120_TRANSPORT_TABLES, default ~/.deep_gemm/sm120_transport_tables.json."""
    return os.environ.get("DG_SM120_TRANSPORT_TABLES",
                          os.path.join(os.path.expanduser("~"), ".deep_gemm", "sm120_transport_tables.json"))


def topology_digest(pair_ok: Any, gin_available: bool, hosts: Sequence[str]) -> str:
    world = len(hosts)
    m = normalize_pair_ok(world, pair_ok)
    reach = "".join("1" if m[(a, b)] else "0" for a in range(world) for b in range(a + 1, world))
    host_ids = {h: i for i, h in enumerate(dict.fromkeys(hosts))}
    return f"hosts={','.join(str(host_ids[h]) for h in hosts)};lsa={reach};gin={int(bool(gin_available))}"


def deployment_key(key: Mapping[str, Any]) -> str:
    return json.dumps(dict(key), sort_keys=True)


def lookup_tuned(key: Mapping[str, Any], path: str | None = None) -> dict[str, Any] | None:
    try:
        with open(path or tuned_table_path(), encoding="utf-8") as f:
            return json.load(f).get(deployment_key(key))
    except (OSError, ValueError):
        return None


def record_tuned(key: Mapping[str, Any], table: TransportTable, report: Mapping[str, Any] | None = None,
                 path: str | None = None) -> str:
    """Rank 0 of an offline tuning run: store `table` for this deployment (read back by transport 'tuned')."""
    path = path or tuned_table_path()
    data: dict[str, Any] = {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        pass
    data[deployment_key(key)] = {"table_name": table.name, "transport_table": list(table.cells),
                                 "candidates": [{"name": c["name"], "ms": c["ms"]} for c in report["candidates"]]
                                 if report else None}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path
