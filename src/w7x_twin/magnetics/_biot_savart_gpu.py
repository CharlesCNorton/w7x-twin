"""GPU Biot-Savart summation as a separate process over npz files:
python -m w7x_twin.magnetics._biot_savart_gpu <input.npz> <output.npz>"""

from __future__ import annotations

import sys

import numpy as np

MU0 = 4.0e-7 * np.pi


def main() -> int:
    import torch

    payload = np.load(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = torch.as_tensor(payload["position"], dtype=torch.float32, device=device)
    moment = torch.as_tensor(payload["moment"], dtype=torch.float32, device=device)
    points = payload["points"]
    # Volume elements have finite extent, so the kernel is regularised at that scale;
    # without it a grid point landing on an element sees an unbounded 1/r^2.
    softening = float(payload["softening"]) if "softening" in payload else 1e-6

    # Both axes of the pairwise tensor are blocked, so memory is bounded by their
    # product rather than by the source count.
    budget = 3.0e7
    target_block = 2048
    source_block = max(1024, int(budget // target_block))
    out = np.empty_like(points, dtype=np.float32)

    for start in range(0, len(points), target_block):
        target = torch.as_tensor(
            points[start : start + target_block], dtype=torch.float32, device=device
        )
        accumulator = torch.zeros(
            (target.shape[0], 3), dtype=torch.float32, device=device
        )
        for first in range(0, len(source), source_block):
            chunk_source = source[first : first + source_block]
            chunk_moment = moment[first : first + source_block]
            delta = target[:, None, :] - chunk_source[None, :, :]
            weight = torch.linalg.norm(delta, dim=-1).clamp_min(softening).pow(-3)
            contribution = torch.linalg.cross(
                chunk_moment.expand(delta.shape[0], -1, -1), delta, dim=-1
            )
            accumulator += (contribution * weight[..., None]).sum(dim=1)
            del delta, weight, contribution
        out[start : start + target_block] = accumulator.cpu().numpy()

    np.savez(sys.argv[2], field=out.astype(np.float64) * (MU0 / (4.0 * np.pi)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
