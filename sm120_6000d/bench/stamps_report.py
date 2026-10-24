# stamps_report.py: per-rank phase breakdown of the stamped EP kernel calls (stamps_h20shape_M*.pt)
import sys, torch
for M in sys.argv[1:]:
    alls = torch.load(f"stamps_h20shape_M{M}.pt")
    starts = [a.view(-1, 16)[:, 0].double().min().item() for a in alls]
    g0 = min(starts)
    print(f"== M{M}: rank start offsets vs earliest rank (us): {[round((s - g0) / 1e3, 1) for s in starts]}")
    print("rank | route-push dispatch-done recv-flag-sent | route-ok map | rows-arrived(med/max) | FC1-end FC2-start comb-flag-rel work-end | comb-all-in end | items/CTA L1wait/CTA")
    for r, a in enumerate(alls):
        s = a.view(-1, 16).double()
        t0 = s[:, 0].min()
        rel = lambda c: (s[:, c] - t0) / 1e3
        nz = lambda c: rel(c)[s[:, c] > 0]
        mx = lambda c: nz(c).max().item() if nz(c).numel() else float("nan")
        mn = lambda c: nz(c).min().item() if nz(c).numel() else float("nan")
        md = lambda c: nz(c).median().item() if nz(c).numel() else float("nan")
        print(f"r{r} | {mx(8):5.1f} {mx(1):5.1f} {mx(12):5.1f} | {md(9):5.1f} {md(10):5.1f} | {md(2):5.1f}/{mx(2):5.1f} | "
              f"{mx(13):5.1f} {mn(14):5.1f} {mx(15):5.1f} {mx(3):5.1f} | {md(4):5.1f} {mx(5):5.1f} | {s[:, 7].mean():.2f} {s[:, 6].mean() / 1e3:.1f}")
