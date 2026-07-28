"""SPEC stepped-pressure equilibria [Hudson et al., Phys. Plasmas 19, 112502 (2012)],
built from a converged VMEC solution so the two solve the same plasma."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import numpy as np
import vmecpp

MU0 = 4.0e-7 * np.pi

#: SPEC works in units where the permeability is one, so its pressure input is mu0 times
#: the pressure in pascals. Supplying pascals directly leaves the plasma beta larger than
#: the equilibrium's by 1/mu0 and the force balance with nothing to converge to.
PRESSURE_TO_SPEC = MU0


@dataclasses.dataclass
class SteppedInput:
    """A SPEC input file and the equilibrium it was built from."""

    path: Path
    extension: str
    num_volumes: int
    interface_transforms: tuple[float, ...]
    interface_flux: tuple[float, ...]
    pressures_pa: tuple[float, ...]
    toroidal_flux_wb: float


def boundary_harmonics(
    wout, mpol: int, ntor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary Rbc(n, m) and Zbs(n, m) with n rescaled to SPEC's per-period indexing."""
    xm = np.asarray(wout.xm).astype(int)
    xn = np.asarray(wout.xn).astype(int)
    nfp = int(wout.nfp)
    surface = int(wout.ns) - 1
    rmnc = np.asarray(wout.rmnc)[:, surface]
    zmns = np.asarray(wout.zmns)[:, surface]

    rbc = np.zeros((2 * ntor + 1, mpol + 1))
    zbs = np.zeros((2 * ntor + 1, mpol + 1))
    for index in range(len(xm)):
        m = int(xm[index])
        n = int(xn[index] // nfp)
        if m > mpol or abs(n) > ntor:
            continue
        rbc[n + ntor, m] = rmnc[index]
        zbs[n + ntor, m] = zmns[index]
    return rbc, zbs


def surface_harmonics(
    wout, flux: float, mpol: int, ntor: int
) -> dict[tuple[int, int], tuple[float, float]]:
    """Surface Rbc and Zbs at one normalised flux, radially interpolated, keyed (m, n) per period."""
    xm = np.asarray(wout.xm).astype(int)
    xn = np.asarray(wout.xn).astype(int)
    nfp = int(wout.nfp)
    rmnc = np.asarray(wout.rmnc)
    zmns = np.asarray(wout.zmns)
    grid = np.linspace(0.0, 1.0, rmnc.shape[1])

    out: dict[tuple[int, int], tuple[float, float]] = {}
    for index in range(len(xm)):
        m = int(xm[index])
        n = int(xn[index] // nfp)
        if m > mpol or abs(n) > ntor:
            continue
        out[(m, n)] = (
            float(np.interp(flux, grid, rmnc[index])),
            float(np.interp(flux, grid, zmns[index])),
        )
    return out


def interface_block(
    wout, flux: np.ndarray, mpol: int, ntor: int
) -> list[str]:
    """Interface geometry block for ``Linitialize`` zero, seeding the Newton with VMEC's own surfaces."""
    surfaces = [surface_harmonics(wout, float(value), mpol, ntor) for value in flux]
    modes = sorted({key for surface in surfaces for key in surface})
    lines = []
    for m, n in modes:
        entries = []
        for surface in surfaces:
            r, z = surface.get((m, n), (0.0, 0.0))
            # R cosine, Z sine, R sine, Z cosine. The last two vanish under stellarator
            # symmetry, which both codes assume here.
            entries.extend([r, z, 0.0, 0.0])
        lines.append(
            f"{m:4d} {n:4d} " + " ".join(f"{value: .15E}" for value in entries)
        )
    return lines


def axis_harmonics(wout, ntor: int) -> tuple[np.ndarray, np.ndarray]:
    """Magnetic axis ``Rac(n)`` and ``Zas(n)`` in SPEC's indexing."""
    xm = np.asarray(wout.xm).astype(int)
    xn = np.asarray(wout.xn).astype(int)
    nfp = int(wout.nfp)
    rmnc = np.asarray(wout.rmnc)[:, 0]
    zmns = np.asarray(wout.zmns)[:, 0]

    rac = np.zeros(ntor + 1)
    zas = np.zeros(ntor + 1)
    for index in range(len(xm)):
        if int(xm[index]) != 0:
            continue
        n = int(xn[index] // nfp)
        if 0 <= n <= ntor:
            rac[n] = rmnc[index]
            zas[n] = zmns[index]
    return rac, zas


#: Namelist blocks that do not depend on the equilibrium.
NUMERIC_BLOCK = """&numericlist
 Linitialize = {initialise:9d}
 LautoinitBn =         0
 Lzerovac    =         0
 Ndiscrete   =         2
 Nquad       =        -1
 iMpol       =        -4
 iNtor       =        -4
 Lsparse     =         0
 Lsvdiota    =         0
 imethod     =         3
 iorder      =         2
 iprecon     =         1
 iotatol     =  -1.000000000000000E+00
 Lextrap     =         0
 Mregular    = {regular:9d}
/
&locallist
 LBeltrami   =         4
 Linitgues   =         1
 Lmatsolver  =         1
/
&globallist
 Lfindzero   =         2
 escale      =   0.000000000000000E+00
 opsilon     =   1.000000000000000E+00
 pcondense   =   4.000000000000000E+00
 epsilon     =   1.000000000000000E+00
 wpoloidal   =   1.000000000000000E+00
 upsilon     =   1.000000000000000E+00
 forcetol    =   1.000000000000000E-10
 c05xmax     = {position_tolerance: .15E}
 c05xtol     =   1.000000000000000E-12
 c05factor   = {newton_step: .15E}
 LreadGF     =         F
 mfreeits    =         0
/
&diagnosticslist
 odetol      =   1.000000000000000E-07
 nPpts       =      {points}
 nPtrj       = {trajectories}
 LHevalues   =         F
 LHevectors  =         F
 LHmatrix    =         F
 Lperturbed  =         0
 dpp         =        -1
 dqq         =        -1
 Lcheck      =         0
 Ltiming     =         F
/
&screenlist
/
"""


def write_input(
    output: vmecpp.VmecOutput,
    directory: str | Path,
    extension: str,
    interface_flux: tuple[float, ...],
    # Raising these to the resolution of the VMEC solve the boundary comes from was tried
    # and measured: at Mpol = Ntor = 8 with Lrad = 12 the zero-pressure force residual is
    # 1.427e-2 against 9.013e-3 here, and one solve costs eighty minutes rather than three.
    # The truncated boundary spectrum is therefore not what holds the residual up.
    mpol: int = 6,
    ntor: int = 6,
    radial_resolution: int = 8,
    poincare_points: int = 400,
    poincare_trajectories: int = 32,
    constraint: int = 1,
    mu: tuple[float, ...] | None = None,
    regular: int = -1,
    initialise: int = 1,
    position_tolerance: float = 1.0e-6,
    newton_step: float = 1.0e-4,
) -> SteppedInput:
    """Write a fixed-boundary SPEC input; ``interface_flux`` places the interior interfaces
    and a bracketed resonance keeps its island inside a volume."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wout = output.wout
    nfp = int(wout.nfp)

    iota_profile = np.asarray(wout.iotaf)
    s_profile = np.linspace(0.0, 1.0, len(iota_profile))
    flux = np.concatenate([np.asarray(interface_flux, dtype=float), [1.0]])
    transforms = tuple(float(value) for value in np.interp(flux, s_profile, iota_profile))
    num_volumes = len(flux)

    # The pressure each volume carries: the mean of the VMEC profile across it, since a
    # stepped profile has one value per volume rather than a continuous one.
    s = np.linspace(0.0, 1.0, len(np.asarray(wout.presf)))
    profile = np.asarray(wout.presf)
    edges = np.concatenate([[0.0], flux])
    volume_pressure = np.array(
        [
            float(np.mean(profile[(s >= edges[i]) & (s <= edges[i + 1])]))
            if np.any((s >= edges[i]) & (s <= edges[i + 1]))
            else float(np.interp(0.5 * (edges[i] + edges[i + 1]), s, profile))
            for i in range(num_volumes)
        ]
    )

    # Parallel current per volume. Under the transform constraint SPEC solves for these,
    # so the values written are only a starting point; without it they are the physics,
    # and zero is a force-free field carrying no net parallel current.
    volume_mu = (
        np.zeros(num_volumes)
        if mu is None
        else np.asarray(mu, dtype=float)[:num_volumes]
    )
    if len(volume_mu) != num_volumes:
        raise ValueError(f"mu has {len(volume_mu)} entries for {num_volumes} volumes")

    rbc, zbs = boundary_harmonics(wout, mpol, ntor)
    rac, zas = axis_harmonics(wout, ntor)
    # SPEC's converging stellarator cases run a positive toroidal flux with positive
    # transform targets. The sign carries no physics for a stellarator-symmetric
    # fixed-boundary solve, and VMEC reports its transform positive whichever way its
    # flux points, so a signed phiedge pairs positive iota targets with a negative flux
    # and asks the volume solves for a field that cannot satisfy both.
    phiedge = abs(float(np.asarray(wout.phi)[-1]))

    lines = ["&physicslist"]
    lines.append(" Igeometry   =         3")
    lines.append(" Istellsym   =         1")
    lines.append(" Lfreebound  =         0")
    lines.append(f" phiedge     = {phiedge: .15E}")
    # The net current rides in the flux-aligned frame, so it flips with the flux sign.
    flux_sign = 1.0 if float(np.asarray(wout.phi)[-1]) >= 0.0 else -1.0
    lines.append(f" curtor      = {flux_sign * float(wout.ctor): .15E}")
    lines.append(" curpol      =   0.000000000000000E+00")
    lines.append(" gamma       =   0.000000000000000E+00")
    lines.append(f" Nfp         = {nfp:9d}")
    lines.append(f" Nvol        = {num_volumes:9d}")
    lines.append(f" Mpol        = {mpol:9d}")
    lines.append(f" Ntor        = {ntor:9d}")
    lines.append(" Lrad        = " + " ".join(f"{radial_resolution:23d}" for _ in range(num_volumes)))
    lines.append(" tflux       = " + " ".join(f"{value: .15E}" for value in flux))
    # Starting guesses for the poloidal flux each volume carries, which the transform
    # constraint's inner solve adjusts. The innermost volume carries none by convention;
    # outward each adds the transform integrated across the volume, which is exact for a
    # vacuum field and close for anything near one.
    pflux = np.zeros(num_volumes)
    for volume in range(1, num_volumes):
        pflux[volume] = pflux[volume - 1] + 0.5 * (
            transforms[volume - 1] + transforms[volume]
        ) * (flux[volume] - flux[volume - 1])
    lines.append(" pflux       = " + " ".join(f"{value: .15E}" for value in pflux))
    lines.append(" helicity    = " + " ".join(" 0.000000000000000E+00" for _ in flux))
    lines.append(f" pscale      = {PRESSURE_TO_SPEC: .15E}")
    lines.append(" Ladiabatic  =         0")
    lines.append(" pressure    = " + " ".join(f"{value: .15E}" for value in volume_pressure))
    lines.append(" adiabatic   = " + " ".join(" 1.000000000000000E+00" for _ in flux))
    lines.append(
        " mu          = " + " ".join(f"{value: .15E}" for value in volume_mu)
    )
    # Constrain the transform at every interface, which is what anchors the solution to
    # the equilibrium it is built from. Under this constraint SPEC runs an inner Newton
    # for the parallel current and the poloidal flux of each volume, and that inner solve
    # failing is one of the two ways the outer force balance can stall; the other is the
    # geometry, so the constraint is a parameter rather than a fixed choice.
    lines.append(f" Lconstraint = {constraint:9d}")
    lines.append(" pl          = " + " ".join(f"{0:23d}" for _ in range(num_volumes + 1)))
    lines.append(" ql          = " + " ".join(f"{0:23d}" for _ in range(num_volumes + 1)))
    lines.append(" pr          = " + " ".join(f"{0:23d}" for _ in range(num_volumes + 1)))
    lines.append(" qr          = " + " ".join(f"{0:23d}" for _ in range(num_volumes + 1)))
    axis_iota = float(np.asarray(wout.iotaf)[0])
    lines.append(
        " iota        = " + " ".join(f"{value: .15E}" for value in (axis_iota, *transforms))
    )
    lines.append(" lp          = " + " ".join(f"{0:23d}" for _ in range(num_volumes + 1)))
    lines.append(" lq          = " + " ".join(f"{0:23d}" for _ in range(num_volumes + 1)))
    lines.append(" rp          = " + " ".join(f"{0:23d}" for _ in range(num_volumes + 1)))
    lines.append(" rq          = " + " ".join(f"{0:23d}" for _ in range(num_volumes + 1)))
    lines.append(
        " oita        = " + " ".join(f"{value: .15E}" for value in (axis_iota, *transforms))
    )
    lines.append(" mupftol     =   1.000000000000000E-12")
    lines.append(" mupfits     =       128")
    lines.append(" Rac         = " + " ".join(f"{value: .15E}" for value in rac))
    lines.append(" Zas         = " + " ".join(f"{value: .15E}" for value in zas))
    lines.append(" Ras         = " + " ".join(" 0.000000000000000E+00" for _ in rac))
    lines.append(" Zac         = " + " ".join(" 0.000000000000000E+00" for _ in zas))

    for m in range(mpol + 1):
        for n in range(-ntor, ntor + 1):
            r = rbc[n + ntor, m]
            z = zbs[n + ntor, m]
            if r == 0.0 and z == 0.0:
                continue
            lines.append(
                f"Rbc({n:3d},{m:3d}) = {r: .15E} Zbs({n:3d},{m:3d}) = {z: .15E} "
                f"Rbs({n:3d},{m:3d}) =  0.000000000000000E+00 "
                f"Zbc({n:3d},{m:3d}) =  0.000000000000000E+00"
            )
    lines.append("/")
    lines.append(
        NUMERIC_BLOCK.format(
            points=poincare_points,
            regular=regular,
            initialise=initialise,
            position_tolerance=position_tolerance,
            newton_step=newton_step,
            trajectories=" ".join(
                f"{poincare_trajectories:d}" for _ in range(num_volumes)
            ),
        )
    )
    if initialise <= 0:
        lines.extend(interface_block(wout, flux, mpol, ntor))

    path = directory / f"{extension}.sp"
    path.write_text("\n".join(lines) + "\n")
    return SteppedInput(
        path=path,
        extension=extension,
        num_volumes=num_volumes,
        interface_transforms=tuple(transforms),
        interface_flux=tuple(float(value) for value in flux),
        pressures_pa=tuple(float(value) for value in volume_pressure),
        toroidal_flux_wb=phiedge,
    )


def run(
    written: SteppedInput, executable: str | Path, timeout_s: float = 7200.0
) -> subprocess.CompletedProcess:
    """Run SPEC on a written input, in its own directory."""
    return subprocess.run(
        [str(executable), written.extension],
        cwd=written.path.parent,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def force_residual(path: str | Path) -> float:
    """The force imbalance the solution ended on, or NaN for an unreadable file."""
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            return float(np.asarray(handle["output/ForceErr"])[0])
    except (OSError, KeyError):
        return float("nan")


def scale_pressure(source: str | Path, destination: str | Path, factor: float) -> Path:
    """Copy an input with its pressure scaled, keeping everything else."""
    source, destination = Path(source), Path(destination)
    lines = []
    for line in source.read_text().splitlines():
        if line.strip().startswith("pscale"):
            lines.append(f" pscale      = {PRESSURE_TO_SPEC * factor: .15E}")
        else:
            lines.append(line)
    destination.write_text("\n".join(lines) + "\n")
    return destination


def continuation(
    written: SteppedInput,
    executable: str | Path,
    factors: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
    verbose: bool = True,
) -> list[dict]:
    """Solve up a pressure ramp, each step restarted from the previous converged geometry."""
    directory = written.path.parent
    steps: list[dict] = []
    current = written.path
    for index, factor in enumerate(factors):
        extension = f"{written.extension}_p{index}"
        scale_pressure(current, directory / f"{extension}.sp", factor)
        completed = subprocess.run(
            [str(executable), extension],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        output = directory / f"{extension}.sp.h5"
        residual = force_residual(output) if output.exists() else float("nan")
        steps.append(
            {
                "factor": factor,
                "extension": extension,
                "returncode": completed.returncode,
                "force_residual": residual,
                "output": str(output),
            }
        )
        if verbose:
            print(
                f"  pressure x {factor:.2f}: exit {completed.returncode}, "
                f"force residual {residual:.3e}"
            )
        # The converged geometry becomes the next step's starting point.
        ended = directory / f"{extension}.sp.end"
        if ended.exists():
            current = ended
        elif not output.exists():
            break
    return steps


def poincare_section(path: str | Path) -> dict:
    """Read the Poincare section SPEC wrote, as (R, Z) per traced trajectory."""
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle["poincare"]
        return {
            "R": np.asarray(group["R"]),
            "Z": np.asarray(group["Z"]),
            "transform": np.asarray(handle["transform"]["fiota"])
            if "transform" in handle
            else None,
        }


def net_winding(r: np.ndarray, z: np.ndarray, axis_r: float, axis_z: float) -> float:
    """Axis-encirclement rate of a trajectory's Poincare points: order iota on a surface, zero librating."""
    angle = np.unwrap(np.arctan2(np.asarray(z) - axis_z, np.asarray(r) - axis_r))
    if angle.size < 2:
        return float("nan")
    return float(abs(angle[-1] - angle[0]) / (2.0 * np.pi * (angle.size - 1)))


def winding_profile(
    r: np.ndarray, z: np.ndarray, axis_r: float, axis_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean radius and (possibly aliased, still monotone) winding number per trajectory, ordered outward."""
    radius = np.array(
        [float(np.mean(np.hypot(r[k] - axis_r, z[k] - axis_z))) for k in range(r.shape[0])]
    )
    winding = np.array(
        [net_winding(r[k], z[k], axis_r, axis_z) for k in range(r.shape[0])]
    )
    order = np.argsort(radius)
    return radius[order], winding[order]


def island_at_resonance(
    r: np.ndarray,
    z: np.ndarray,
    axis_r: float,
    axis_z: float,
    resonance: float,
    tolerance: float = 1.0e-3,
) -> dict:
    """Island width at a known resonance from the winding-locked run, alias tried both ways, in metres."""
    radius, winding = winding_profile(r, z, axis_r, axis_z)
    good = np.isfinite(winding) & np.isfinite(radius)
    radius, winding = radius[good], winding[good]
    if radius.size < 3:
        return {"width_m": 0.0, "trajectories": 0, "winding": float("nan")}

    best = {"width_m": 0.0, "trajectories": 0, "winding": float("nan")}
    for target, implied in ((resonance, resonance), (1.0 - resonance, resonance)):
        near = np.abs(winding - target) < tolerance
        if not near.any():
            continue
        indices = np.flatnonzero(near)
        # The longest contiguous run, so a stray trajectory elsewhere is not counted in.
        splits = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        run = max(splits, key=len)
        if len(run) < 2:
            continue
        width = float(radius[run[-1]] - radius[run[0]])
        if width > best["width_m"]:
            best = {
                "width_m": width,
                "trajectories": int(len(run)),
                "winding": float(np.mean(winding[run])),
                "implied_transform": float(implied),
                "inner_m": float(radius[run[0]]),
                "outer_m": float(radius[run[-1]]),
            }
    return best


def island_width(
    r: np.ndarray,
    z: np.ndarray,
    axis_r: float,
    axis_z: float,
    flatness: float = 0.1,
    minimum_trajectories: int = 3,
) -> dict:
    """Width of the widest island as the radial extent of the winding-number plateau, in metres."""
    radius, winding = winding_profile(r, z, axis_r, axis_z)
    good = np.isfinite(winding) & np.isfinite(radius)
    radius, winding = radius[good], winding[good]
    if radius.size < minimum_trajectories + 2:
        return {"width_m": 0.0, "trajectories": 0, "winding": float("nan")}

    steps = np.abs(np.diff(winding))
    typical = float(np.median(steps))
    if typical <= 0.0:
        return {"width_m": 0.0, "trajectories": 0, "winding": float("nan")}
    flat = steps < flatness * typical

    best = {"width_m": 0.0, "trajectories": 0, "winding": float("nan")}
    start = None
    for index in range(len(flat) + 1):
        if index < len(flat) and flat[index]:
            start = index if start is None else start
            continue
        if start is not None:
            stop = index
            count = stop - start + 1
            width = float(radius[stop] - radius[start])
            if count >= minimum_trajectories and width > best["width_m"]:
                best = {
                    "width_m": width,
                    "trajectories": int(count),
                    "winding": float(np.mean(winding[start : stop + 1])),
                    "inner_m": float(radius[start]),
                    "outer_m": float(radius[stop]),
                }
            start = None
    return best
