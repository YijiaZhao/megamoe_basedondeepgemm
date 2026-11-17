# fp4_index_logits_grouped (the grouped verify indexer of sglang_v0.5.22_dsv41.patch) against the stock path, the
# [rows, L] slot gather + fp4_index_logits_decode: bit-exact logits on the verify layout (16 requests x 6 rows) with
# equal and uneven context lengths, then cold-L2 time per call. L is the CUDA graph capacity (180K context), ~100K live.
#   docker run --rm --gpus device=0 -v $PWD:/s --entrypoint bash lmsysorg/sglang:v0.5.22-cu130 -c \
#     "cd /sgl-workspace/sglang && git apply /s/sglang_v0.5.22_dsv41.patch && python3 /s/test_indexer_grouped.py"
import torch
import triton

from sglang.kernels.ops.attention.dsv4 import fp4_indexer as fi

H, D, PS = 32, 128, 64  # index heads, index head dim, index-K pool page size
NREQ, G, CTX, CAP = 16, 6, 100_000, 180_000
dev = 'cuda'
torch.manual_seed(0)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def bench(fn, n=30):
    fn()
    torch.cuda.synchronize()
    times = []
    for i in range(n):
        flush.fill_(i)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return sorted(times)[n // 2]


for ratio in (2, 1):
    lmax = per = CAP // ratio
    npages = triton.cdiv(NREQ * per + PS, PS) + 1
    table = torch.randint(0, 256, (npages, PS * 68), dtype=torch.uint8, device=dev)
    table[:, PS * 64:] = torch.randint(118, 126, (npages, PS * 4), dtype=torch.uint8, device=dev)
    # each request's compressed positions on shuffled pages: token t * ratio holds pool slot * ratio
    perm = torch.randperm(npages - 1, device=dev)
    req_table = torch.zeros(NREQ, CAP, dtype=torch.int32, device=dev)
    for r in range(NREQ):
        c = torch.arange(per, device=dev) + r * per
        req_table[r, torch.arange(per, device=dev) * ratio] = ((perm[c // PS] * PS + c % PS) * ratio).int()
    req = torch.arange(NREQ, device=dev).repeat_interleave(G)
    pos = (CTX - G + torch.arange(G, device=dev)).repeat(NREQ)
    q = (torch.randn(NREQ * G, H, D, device=dev) * 0.5).bfloat16()
    w = (torch.rand(NREQ * G, H, device=dev) * 0.1).bfloat16()

    def stock(lens):
        j = torch.arange(lmax, device=dev)
        slots = req_table[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
        slots = slots.masked_fill(~(j[None, :] < lens[:, None]), 0)
        return fi.fp4_index_logits_decode(q, w, slots, lens, table, PS)

    def grouped(lens):
        return fi.fp4_index_logits_grouped(q, w, req, req_table, lens, table, PS, lmax, ratio, G)

    def same(lens):
        a, b = stock(lens), grouped(lens)
        live = torch.arange(lmax, device=dev)[None, :] < lens[:, None]
        return torch.equal(a[live], b[live]) and bool((b[~live] == float('-inf')).all())

    lens = ((pos + 1) // ratio).to(torch.int64)
    uneven = lens.clone()
    uneven[G:2 * G] = torch.arange(G, device=dev) + 30_000 // ratio  # a shorter request
    assert same(lens) and same(uneven), f'ratio {ratio}: grouped logits differ from the stock path'
    print(f'ratio {ratio}: bit-exact; stock {bench(lambda: stock(lens)):7.1f} us, '
          f'grouped {bench(lambda: grouped(lens)):7.1f} us per call', flush=True)
