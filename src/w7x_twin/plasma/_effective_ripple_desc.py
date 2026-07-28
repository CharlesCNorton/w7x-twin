"""Nemov effective ripple via DESC's bounce averaging, as a separate process:
python -m w7x_twin.plasma._effective_ripple_desc <wout.nc> <output.npz> [num_surfaces]"""

from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    from desc.grid import LinearGrid
    from desc.integrals import Bounce2D
    from desc.vmec import VMECIO

    wout_path = sys.argv[1]
    out_path = sys.argv[2]
    num_surfaces = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    equilibrium = VMECIO.load(wout_path)

    # The axis and the boundary are excluded: the bounce integrals need a well-defined
    # trapped region, which neither limit provides.
    rho = np.linspace(0.15, 0.95, num_surfaces)
    alpha = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    grid = LinearGrid(
        rho=rho,
        M=equilibrium.M_grid,
        N=equilibrium.N_grid,
        NFP=equilibrium.NFP,
        sym=False,
    )

    num_transit = 20
    data = equilibrium.compute(
        "effective ripple 3/2",
        grid=grid,
        angle=Bounce2D.angle(equilibrium, X=16, Y=32, rho=rho),
        alpha=alpha,
        Y_B=128,
        num_transit=num_transit,
        num_well=20 * num_transit,
        num_quad=32,
        num_pitch=64,
    )
    eps_32 = np.asarray(grid.compress(data["effective ripple 3/2"]))

    np.savez(
        out_path,
        rho=rho,
        s=rho**2,
        eps_32=eps_32,
        eps_eff=eps_32 ** (2.0 / 3.0),
    )
    for r, value in zip(rho, eps_32, strict=True):
        print(f"  rho={r:.3f}  s={r**2:.3f}  eps_eff = {value ** (2/3):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
