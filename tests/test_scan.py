import torch

from ingram.scan import segmented_scan


def test_scan_matches_sequential():
    torch.manual_seed(0)
    lengths = [1, 5, 2, 9, 1, 33]
    E, d = sum(lengths), 4
    a = torch.rand(E, 1, dtype=torch.float64)
    b = torch.randn(E, d, dtype=torch.float64)
    seg_start = torch.cat([torch.full((n,), s) for n, s in zip(lengths, [sum(lengths[:i]) for i in range(len(lengths))])])
    out = segmented_scan(a.expand(E, d), b, seg_start, max(lengths))
    ref, j = torch.zeros_like(b), 0
    for n in lengths:
        state = torch.zeros(d, dtype=torch.float64)
        for _ in range(n):
            state = a[j] * state + b[j]
            ref[j] = state
            j += 1
    assert torch.allclose(out, ref, atol=1e-12)
