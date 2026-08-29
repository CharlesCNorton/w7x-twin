"""Field of the currents flowing in the plasma, by volume Biot-Savart integration."""

from __future__ import annotations

import dataclasses
import numpy as np
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from w7x_twin.magnetics.field import FieldGrid

if TYPE_CHECKING:
    import vmecpp


# -- from plasma_response ---------------------------------------------------------

MU0 = 4.0e-7 * np.pi


@dataclasses.dataclass
class CurrentDistribution:
    """Plasma current density sampled over the plasma volume."""

    position: np.ndarray  # (n, 3) Cartesian
    moment: np.ndarray  # (n, 3) J dV, in ampere metres

    @property
    def num_sources(self) -> int:
        return self.position.shape[0]

    def element_size(self) -> float:
        """Element linear extent, the Biot-Savart softening length."""
        extent = self.position.max(axis=0) - self.position.min(axis=0)
        volume = float(np.prod(extent))
        return float((volume / max(self.num_sources, 1)) ** (1.0 / 3.0))

    def net_toroidal_current_a(self) -> float:
        """Net current through a poloidal plane, the check against ``ctor``."""
        phi = np.arctan2(self.position[:, 1], self.position[:, 0])
        toroidal = -self.moment[:, 0] * np.sin(phi) + self.moment[:, 1] * np.cos(phi)
        radius = np.hypot(self.position[:, 0], self.position[:, 1])
        return float(np.sum(toroidal / radius) / (2.0 * np.pi))


def current_distribution(
    output: vmecpp.VmecOutput,
    num_theta: int = 40,
    num_zeta: int = 96,
    radial_stride: int = 1,
) -> CurrentDistribution:
    """Sample J dV over the plasma, on the half radial grid and the full torus."""
    wout = output.wout
    ns = int(wout.ns)
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
    xm_nyq, xn_nyq = np.asarray(wout.xm_nyq), np.asarray(wout.xn_nyq)

    rmnc = np.asarray(wout.rmnc)
    zmns = np.asarray(wout.zmns)
    curru = np.asarray(wout.currumnc)
    currv = np.asarray(wout.currvmnc)

    surfaces = np.arange(1, ns, radial_stride)
    ds = radial_stride / (ns - 1)

    theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, num_zeta, endpoint=False)
    theta2d, zeta2d = np.meshgrid(theta, zeta, indexing="ij")
    dtheta = 2.0 * np.pi / num_theta
    dzeta = 2.0 * np.pi / num_zeta

    angle = (
        xm[None, None, :] * theta2d[:, :, None] - xn[None, None, :] * zeta2d[:, :, None]
    )
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    angle_nyq = (
        xm_nyq[None, None, :] * theta2d[:, :, None]
        - xn_nyq[None, None, :] * zeta2d[:, :, None]
    )
    cos_nyq = np.cos(angle_nyq)

    cos_z, sin_z = np.cos(zeta2d), np.sin(zeta2d)

    positions = []
    moments = []
    for surface in surfaces:
        # Geometry on the half grid, between the two bracketing full-grid surfaces.
        r_coeff = 0.5 * (rmnc[:, surface] + rmnc[:, surface - 1])
        z_coeff = 0.5 * (zmns[:, surface] + zmns[:, surface - 1])

        r = cos_a @ r_coeff
        z = sin_a @ z_coeff
        dr_dtheta = -(sin_a * xm[None, None, :]) @ r_coeff
        dr_dzeta = (sin_a * xn[None, None, :]) @ r_coeff
        dz_dtheta = (cos_a * xm[None, None, :]) @ z_coeff
        dz_dzeta = -(cos_a * xn[None, None, :]) @ z_coeff

        # VMEC stores the current density already multiplied by the Jacobian, and its
        # coordinate system is left-handed, so the volume element is absorbed and one
        # overall sign remains. Both are fixed by requiring the net toroidal current
        # to reproduce ``ctor``; the on-axis diamagnetic field then follows.
        j_theta = -(cos_nyq @ curru[:, surface])
        j_zeta = -(cos_nyq @ currv[:, surface])

        e_theta = np.stack(
            [dr_dtheta * cos_z, dr_dtheta * sin_z, dz_dtheta], axis=-1
        )
        e_zeta = np.stack(
            [dr_dzeta * cos_z - r * sin_z, dr_dzeta * sin_z + r * cos_z, dz_dzeta],
            axis=-1,
        )

        current = (
            j_theta[..., None] * e_theta + j_zeta[..., None] * e_zeta
        ) * (ds * dtheta * dzeta)

        positions.append(np.stack([r * cos_z, r * sin_z, z], axis=-1).reshape(-1, 3))
        moments.append(current.reshape(-1, 3))

    return CurrentDistribution(
        position=np.concatenate(positions), moment=np.concatenate(moments)
    )


@dataclasses.dataclass
class Convergence:
    """Field at a set of points under successive refinement of the volume sampling."""

    samplings: list[tuple[int, int, int]]
    sources: list[int]
    field: list[np.ndarray]

    def changes(self) -> list[float]:
        """Largest change in |B| between successive refinements, in tesla."""
        out = []
        for coarse, fine in zip(self.field, self.field[1:], strict=False):
            out.append(
                float(
                    np.max(
                        np.abs(
                            np.linalg.norm(fine, axis=-1) - np.linalg.norm(coarse, axis=-1)
                        )
                    )
                )
            )
        return out

    def converged(self, tolerance: float) -> bool:
        changes = self.changes()
        return bool(changes and changes[-1] <= tolerance)

    def bound(self) -> float:
        """Residual the finest sampling carries, taken as its last change."""
        changes = self.changes()
        return changes[-1] if changes else float("nan")


def refine_until_converged(
    output: vmecpp.VmecOutput,
    points: np.ndarray,
    samplings: tuple[tuple[int, int, int], ...] = (
        (40, 120, 8), (60, 180, 4), (80, 240, 2), (120, 360, 1),
    ),
    tolerance: float = 1.0e-5,
    interpreter: str | None = None,
    verbose: bool = True,
) -> Convergence:
    """Plasma field at ``points`` over refining volume samplings, each result carrying its residual."""
    record = Convergence(samplings=list(samplings), sources=[], field=[])
    for num_theta, num_zeta, stride in samplings:
        distribution = current_distribution(
            output, num_theta=num_theta, num_zeta=num_zeta, radial_stride=stride
        )
        value = (
            field_at_points_gpu(distribution, points, interpreter)
            if interpreter
            else field_at_points(distribution, points)
        )
        record.sources.append(distribution.num_sources)
        record.field.append(value)
        if verbose:
            changes = record.changes()
            note = f", change {changes[-1]:.3e} T" if changes else ""
            print(
                f"  {num_theta}x{num_zeta}, stride {stride}: "
                f"{distribution.num_sources} elements{note}"
            )
        if record.converged(tolerance):
            break
    return record


@dataclasses.dataclass
class BoundarySheet:
    """The boundary as the virtual-casing sheet current (n x B) dS / mu0."""

    position: np.ndarray  # (n, 3) Cartesian
    moment: np.ndarray  # (n, 3) K dS, in ampere metres

    @property
    def num_sources(self) -> int:
        return self.position.shape[0]

    def net_current_a(self) -> float:
        """Net toroidal current the sheet carries, for checking against ``ctor``."""
        phi = np.arctan2(self.position[:, 1], self.position[:, 0])
        toroidal = -self.moment[:, 0] * np.sin(phi) + self.moment[:, 1] * np.cos(phi)
        radius = np.hypot(self.position[:, 0], self.position[:, 1])
        return float(np.sum(toroidal / radius) / (2.0 * np.pi))


def boundary_sheet(
    output: vmecpp.VmecOutput, num_theta: int = 128, num_zeta: int = 320,
    vacuum=None,
) -> BoundarySheet:
    """Virtual-casing sheet K = n x (B - B_vacuum) / mu0 on the boundary, oriented to carry
    the equilibrium's own net toroidal current."""
    wout = output.wout
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
    xm_nyq, xn_nyq = np.asarray(wout.xm_nyq), np.asarray(wout.xn_nyq)
    rmnc, zmns = np.asarray(wout.rmnc), np.asarray(wout.zmns)
    bsupu, bsupv = np.asarray(wout.bsupumnc), np.asarray(wout.bsupvmnc)

    theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, num_zeta, endpoint=False)
    theta2d, zeta2d = np.meshgrid(theta, zeta, indexing="ij")
    dtheta = 2.0 * np.pi / num_theta
    dzeta = 2.0 * np.pi / num_zeta

    angle = (
        xm[None, None, :] * theta2d[:, :, None] - xn[None, None, :] * zeta2d[:, :, None]
    )
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    angle_nyq = (
        xm_nyq[None, None, :] * theta2d[:, :, None]
        - xn_nyq[None, None, :] * zeta2d[:, :, None]
    )
    cos_nyq = np.cos(angle_nyq)
    cos_z, sin_z = np.cos(zeta2d), np.sin(zeta2d)

    r = cos_a @ rmnc[:, -1]
    z = sin_a @ zmns[:, -1]
    dr_dtheta = -(sin_a * xm[None, None, :]) @ rmnc[:, -1]
    dr_dzeta = (sin_a * xn[None, None, :]) @ rmnc[:, -1]
    dz_dtheta = (cos_a * xm[None, None, :]) @ zmns[:, -1]
    dz_dzeta = -(cos_a * xn[None, None, :]) @ zmns[:, -1]

    # The contravariant field is stored on the half grid, so the boundary value is the
    # linear extrapolation of the two outermost of them.
    edge_u = 1.5 * bsupu[:, -1] - 0.5 * bsupu[:, -2]
    edge_v = 1.5 * bsupv[:, -1] - 0.5 * bsupv[:, -2]
    b_sup_u = cos_nyq @ edge_u
    b_sup_v = cos_nyq @ edge_v

    e_theta = np.stack([dr_dtheta * cos_z, dr_dtheta * sin_z, dz_dtheta], axis=-1)
    e_zeta = np.stack(
        [dr_dzeta * cos_z - r * sin_z, dr_dzeta * sin_z + r * cos_z, dz_dzeta], axis=-1
    )
    field = b_sup_u[..., None] * e_theta + b_sup_v[..., None] * e_zeta
    normal = np.cross(e_theta, e_zeta) * (dtheta * dzeta)

    if vacuum is not None:
        coil_r, coil_phi, coil_z = vacuum(r.ravel(), zeta2d.ravel(), z.ravel())
        shape = r.shape
        coil = np.stack(
            [
                coil_r.reshape(shape) * cos_z - coil_phi.reshape(shape) * sin_z,
                coil_r.reshape(shape) * sin_z + coil_phi.reshape(shape) * cos_z,
                coil_z.reshape(shape),
            ],
            axis=-1,
        )
        field = field - coil

    sheet = np.cross(normal, field) / MU0
    position = np.stack([r * cos_z, r * sin_z, z], axis=-1).reshape(-1, 3)
    result = BoundarySheet(position=position, moment=sheet.reshape(-1, 3))

    # VMEC's angles are left-handed, so the surface element points inward as written. The
    # sign is fixed by the equilibrium's own net toroidal current rather than by the
    # convention, which is the same check the volume sampling is anchored on.
    target = float(wout.ctor)
    if target != 0.0 and result.net_current_a() * target < 0.0:
        result = BoundarySheet(position=position, moment=-result.moment)
    elif target == 0.0 and result.net_current_a() > 0.0:
        result = BoundarySheet(position=position, moment=-result.moment)
    return result


def virtual_casing_field(
    sheet: BoundarySheet, points: np.ndarray, chunk: int = 2048
) -> np.ndarray:
    """Plasma field at exterior points, from the boundary sheet current."""
    return field_at_points(
        CurrentDistribution(position=sheet.position, moment=sheet.moment), points, chunk
    )


def extrapolated_field(
    record: Convergence,
) -> tuple[np.ndarray, np.ndarray]:
    """Richardson extrapolation to zero element size, with the last correction as its residual."""
    if len(record.field) < 3:
        return record.field[-1], np.full(record.field[-1].shape[:-1], np.nan)
    last, middle, first = record.field[-1], record.field[-2], record.field[-3]
    step_one = last - middle
    step_two = middle - first
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(step_two) > 0.0, step_one / step_two, 0.0)
    # A geometric sequence sums to its last step over one minus the ratio. Ratios at or
    # above one are not converging, so the sequence is left where it stands.
    ratio = np.where(np.isfinite(ratio) & (np.abs(ratio) < 0.95), ratio, 0.0)
    correction = step_one * ratio / (1.0 - ratio)
    return last + correction, np.linalg.norm(correction, axis=-1)


def field_at_points(
    currents: CurrentDistribution,
    points: np.ndarray,
    chunk: int = 2048,
    source_chunk: int = 200_000,
) -> np.ndarray:
    """Biot-Savart field of the sampled current, chunked over targets and sources."""
    out = np.zeros_like(points)
    source = currents.position
    moment = currents.moment
    for start in range(0, len(points), chunk):
        target = points[start : start + chunk]
        total = np.zeros_like(target)
        for first in range(0, len(source), source_chunk):
            here = source[first : first + source_chunk]
            weights = moment[first : first + source_chunk]
            delta = target[:, None, :] - here[None, :, :]
            distance = np.linalg.norm(delta, axis=-1)
            weight = 1.0 / np.maximum(distance, 1e-6) ** 3
            total += np.einsum(
                "tsk,ts->tk", np.cross(weights[None, :, :], delta), weight
            )
        out[start : start + chunk] = total
    return out * (MU0 / (4.0 * np.pi))


def field_at_points_gpu(
    currents: CurrentDistribution,
    points: np.ndarray,
    interpreter: str,
    work_dir: str | Path = ".",
    softening: float | None = None,
) -> np.ndarray:
    """The same integral through the GPU worker; ``interpreter`` is a python with CUDA torch."""
    if not Path(interpreter).exists():
        raise FileNotFoundError(
            f"no interpreter at {interpreter}. The volume integral runs in another "
            "process so a CUDA-capable torch is not a dependency of this package; "
            "name one with W7X_TWIN_GPU_PYTHON, or leave it unset to sum on the CPU."
        )
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    request = work_dir / "_biot_savart_in.npz"
    response = work_dir / "_biot_savart_out.npz"

    np.savez(
        request,
        position=currents.position.astype(np.float32),
        moment=currents.moment.astype(np.float32),
        points=points.astype(np.float32),
        softening=np.float64(
            softening if softening is not None else currents.element_size()
        ),
    )
    completed = subprocess.run(
        [interpreter, "-m", "w7x_twin.magnetics._biot_savart_gpu", str(request), str(response)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"the GPU Biot-Savart worker exited {completed.returncode} on "
            f"{currents.num_sources} sources and {len(points)} points.\n"
            f"  {interpreter} -m w7x_twin.magnetics._biot_savart_gpu {request} {response}\n"
            + "\n".join(completed.stderr.strip().splitlines()[-12:])
        )
    if not response.exists():
        raise RuntimeError(
            f"the GPU Biot-Savart worker exited cleanly but wrote no {response.name}"
        )
    field = np.load(response)["field"]
    request.unlink(missing_ok=True)
    response.unlink(missing_ok=True)
    return field


def inside_boundary(
    output: vmecpp.VmecOutput, r: np.ndarray, phi: np.ndarray, z: np.ndarray,
    num_theta: int = 256,
) -> np.ndarray:
    """True inside the boundary, by crossing-number test at the nearest toroidal plane."""
    from w7x_twin.mhd import diagnostics

    from w7x_twin.hardware.walls import inside_contour

    r = np.asarray(r, dtype=float)
    phi = np.mod(np.asarray(phi, dtype=float), 2.0 * np.pi)
    z = np.asarray(z, dtype=float)
    out = np.zeros(r.shape, dtype=bool)
    for angle in np.unique(phi):
        wall_r, wall_z = diagnostics.boundary_cut(output.wout, float(angle), num_theta)
        here = phi == angle
        out[here] = inside_contour(r[here], z[here], wall_r, wall_z)
    return out


def field_on_grid(
    currents: CurrentDistribution,
    grid: FieldGrid,
    chunk: int = 2048,
    verbose: bool = True,
    interpreter: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plasma field on a cylindrical grid, in the layout the coil response uses."""
    period = 2.0 * np.pi / grid.num_field_periods
    phi = np.linspace(0.0, period, grid.num_phi, endpoint=False)
    z = np.linspace(grid.z_min, grid.z_max, grid.num_z)
    r = np.linspace(grid.r_min, grid.r_max, grid.num_r)
    phi3, z3, r3 = np.meshgrid(phi, z, r, indexing="ij")
    points = np.stack(
        [r3 * np.cos(phi3), r3 * np.sin(phi3), z3], axis=-1
    ).reshape(-1, 3)

    if verbose:
        print(
            f"[plasma] {currents.num_sources} current elements onto "
            f"{len(points)} grid points"
        )
    started = time.monotonic()
    cartesian = (
        field_at_points_gpu(currents, points, interpreter)
        if interpreter
        else field_at_points(currents, points, chunk)
    )
    if verbose:
        print(f"[plasma] done in {time.monotonic() - started:.1f} s")

    flat_phi = phi3.ravel()
    b_x, b_y, b_z = cartesian[:, 0], cartesian[:, 1], cartesian[:, 2]
    return (
        b_x * np.cos(flat_phi) + b_y * np.sin(flat_phi),
        -b_x * np.sin(flat_phi) + b_y * np.cos(flat_phi),
        b_z,
    )
