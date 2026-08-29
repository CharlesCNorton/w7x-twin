"""Per-circuit vacuum response tables and the error-field harmonic conventions."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from w7x_twin.hardware.machine import CoilSet, FieldGrid

# The two functions that build a response table import the solver where they call it;
# the contraction and interpolation below read plain arrays, so the tracer stands
# without it installed.
if TYPE_CHECKING:
    import vmecpp

# -- from field -------------------------------------------------------------------

def makegrid_parameters(grid: FieldGrid) -> vmecpp.MakegridParameters:
    import vmecpp

    return vmecpp.MakegridParameters(
        normalize_by_currents=grid.normalize_by_currents,
        assume_stellarator_symmetry=grid.stellarator_symmetric,
        number_of_field_periods=grid.num_field_periods,
        r_grid_minimum=grid.r_min,
        r_grid_maximum=grid.r_max,
        number_of_r_grid_points=grid.num_r,
        z_grid_minimum=grid.z_min,
        z_grid_maximum=grid.z_max,
        number_of_z_grid_points=grid.num_z,
        number_of_phi_grid_points=grid.num_phi,
    )


def full_torus_grid(grid: FieldGrid, phi_points_per_period: int | None = None) -> FieldGrid:
    """The grid over the whole torus with no symmetry assumed, for aperiodic circuits."""
    per_period = phi_points_per_period or grid.num_phi
    return dataclasses.replace(
        grid,
        num_field_periods=1,
        stellarator_symmetric=False,
        num_phi=per_period * grid.num_field_periods,
    )


def _cache_key(coils: CoilSet, grid: FieldGrid) -> str:
    digest = hashlib.sha256(coils.path.read_bytes()).hexdigest()[:12]
    return (
        f"{coils.path.stem}_{digest}_r{grid.num_r}z{grid.num_z}p{grid.num_phi}"
        f"_{grid.r_min:g}-{grid.r_max:g}_{grid.z_min:g}-{grid.z_max:g}"
    )


def build_response_table(
    coils: CoilSet,
    grid: FieldGrid | None = None,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> vmecpp.MagneticFieldResponseTable:
    """Compute the per-circuit vacuum field response, caching it on disk."""
    import vmecpp

    grid = grid or coils.grid
    params = makegrid_parameters(grid)

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"response_{_cache_key(coils, grid)}.npz"
        if cache_path.exists():
            stored = np.load(cache_path)
            if verbose:
                print(f"[field] cache hit {cache_path.name}")
            return vmecpp.MagneticFieldResponseTable(
                parameters=params,
                b_r=stored["b_r"],
                b_p=stored["b_p"],
                b_z=stored["b_z"],
            )

    if verbose:
        print(
            f"[field] Biot-Savart on {grid.num_r}x{grid.num_z}x{grid.num_phi} "
            f"= {grid.num_cells} points for {coils.num_circuits} circuits"
        )
    started = time.monotonic()
    table = vmecpp.MagneticFieldResponseTable.from_coils_file(coils.path, params)
    if verbose:
        print(f"[field] done in {time.monotonic() - started:.1f} s")

    if cache_path is not None:
        np.savez_compressed(
            cache_path, b_r=table.b_r, b_p=table.b_p, b_z=table.b_z
        )
        if verbose:
            print(f"[field] cached {cache_path.name}")
    return table


class VacuumField:
    """One current setting's vacuum field, contracted once for trilinear evaluation."""

    def __init__(
        self, table: vmecpp.MagneticFieldResponseTable, currents: np.ndarray
    ) -> None:
        p = table.parameters
        self.num_r = p.number_of_r_grid_points
        self.num_z = p.number_of_z_grid_points
        self.num_phi = p.number_of_phi_grid_points
        self.r_min, self.r_max = p.r_grid_minimum, p.r_grid_maximum
        self.z_min, self.z_max = p.z_grid_minimum, p.z_grid_maximum
        self.num_field_periods = p.number_of_field_periods
        self.period = 2.0 * np.pi / self.num_field_periods

        weights = np.asarray(currents, dtype=float)
        shape = (self.num_phi, self.num_z, self.num_r)
        # One (component, phi, z, r) block, so every interpolation corner is a single
        # gather over all three components.
        self.b = np.stack(
            [
                (weights @ table.b_r).reshape(shape),
                (weights @ table.b_p).reshape(shape),
                (weights @ table.b_z).reshape(shape),
            ]
        )
        self._digest: str | None = None

        self.dr = (self.r_max - self.r_min) / (self.num_r - 1)
        self.dz = (self.z_max - self.z_min) / (self.num_z - 1)
        self.dphi = self.period / self.num_phi

    @property
    def b_r(self) -> np.ndarray:
        return self.b[0]

    @property
    def b_phi(self) -> np.ndarray:
        return self.b[1]

    @property
    def b_z(self) -> np.ndarray:
        return self.b[2]

    def with_added_field(
        self, b_r: np.ndarray, b_phi: np.ndarray, b_z: np.ndarray
    ) -> VacuumField:
        """A copy carrying this field plus the given one on the same grid."""
        total = copy.copy(self)
        total.b = self.b + np.stack([b_r, b_phi, b_z])
        total._digest = None
        return total

    def digest(self) -> str:
        """Content hash of the contracted field and its grid, for caches keyed on it."""
        if self._digest is None:
            digest = hashlib.sha256(self.b.tobytes())
            digest.update(
                repr(
                    (
                        self.num_r, self.num_z, self.num_phi, self.num_field_periods,
                        self.r_min, self.r_max, self.z_min, self.z_max,
                    )
                ).encode()
            )
            self._digest = digest.hexdigest()[:16]
        return self._digest

    def __call__(
        self,
        r: np.ndarray | float,
        phi: np.ndarray | float,
        z: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trilinearly interpolate (B_r, B_phi, B_z), NaN outside the grid; inputs broadcast."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        phi = np.mod(np.atleast_1d(np.asarray(phi, dtype=float)), self.period)

        fr = (r - self.r_min) / self.dr
        fz = (z - self.z_min) / self.dz
        fp = phi / self.dphi

        inside = (fr >= 0) & (fr <= self.num_r - 1) & (fz >= 0) & (fz <= self.num_z - 1)
        inside &= np.isfinite(fr) & np.isfinite(fz) & np.isfinite(fp)
        # Non-finite coordinates would cast to a garbage index, so park them on a
        # valid cell and discard the result through ``inside`` afterwards.
        fr = np.where(inside, fr, 0.0)
        fz = np.where(inside, fz, 0.0)
        fp = np.where(inside, fp, 0.0)

        ir = np.clip(np.floor(fr), 0, self.num_r - 2).astype(np.intp)
        iz = np.clip(np.floor(fz), 0, self.num_z - 2).astype(np.intp)
        ip = np.floor(fp).astype(np.intp) % self.num_phi
        tr, tz, tp = fr - ir, fz - iz, fp - np.floor(fp)

        ip1 = (ip + 1) % self.num_phi
        acc = np.zeros((3,) + fr.shape)
        for p_index, wp in ((ip, 1 - tp), (ip1, tp)):
            for dz_, wz in ((0, 1 - tz), (1, tz)):
                for dr_, wr in ((0, 1 - tr), (1, tr)):
                    acc += (wp * wz * wr) * self.b[:, p_index, iz + dz_, ir + dr_]
        out = np.where(inside, acc, np.nan)
        return out[0], out[1], out[2]

    def magnitude(self, r, phi, z) -> np.ndarray:
        br, bp, bz = self(r, phi, z)
        return np.sqrt(br * br + bp * bp + bz * bz)

    def with_gradient(
        self,
        r: np.ndarray | float,
        phi: np.ndarray | float,
        z: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(B, grad B) of the trilinear interpolant: components (3, N) ordered
        (R, phi, Z), and exact partials (3, 3, N) as d B_i / d(R, R dphi, Z), from the
        same corner gather the field evaluation uses; NaN outside the grid."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        phi = np.mod(np.atleast_1d(np.asarray(phi, dtype=float)), self.period)

        fr = (r - self.r_min) / self.dr
        fz = (z - self.z_min) / self.dz
        fp = phi / self.dphi

        inside = (fr >= 0) & (fr <= self.num_r - 1) & (fz >= 0) & (fz <= self.num_z - 1)
        inside &= np.isfinite(fr) & np.isfinite(fz) & np.isfinite(fp)
        fr = np.where(inside, fr, 0.0)
        fz = np.where(inside, fz, 0.0)
        fp = np.where(inside, fp, 0.0)

        ir = np.clip(np.floor(fr), 0, self.num_r - 2).astype(np.intp)
        iz = np.clip(np.floor(fz), 0, self.num_z - 2).astype(np.intp)
        ip = np.floor(fp).astype(np.intp) % self.num_phi
        tr, tz, tp = fr - ir, fz - iz, fp - np.floor(fp)
        ip1 = (ip + 1) % self.num_phi

        value = np.zeros((3,) + fr.shape)
        gradient = np.zeros((3, 3) + fr.shape)
        for p_index, wp, sp in ((ip, 1 - tp, -1.0), (ip1, tp, 1.0)):
            for dz_, wz, sz in ((0, 1 - tz, -1.0), (1, tz, 1.0)):
                for dr_, wr, sr in ((0, 1 - tr, -1.0), (1, tr, 1.0)):
                    corner = self.b[:, p_index, iz + dz_, ir + dr_]
                    value += (wp * wz * wr) * corner
                    gradient[:, 0] += (wp * wz * sr / self.dr) * corner
                    gradient[:, 1] += (sp * wz * wr / self.dphi) * corner
                    gradient[:, 2] += (wp * sz * wr / self.dz) * corner
        # The toroidal derivative per radian becomes per arc metre at the point's R.
        gradient[:, 1] /= np.maximum(r, 1e-9)
        value = np.where(inside, value, np.nan)
        gradient = np.where(inside, gradient, np.nan)
        return value, gradient


def field_at(
    table: vmecpp.MagneticFieldResponseTable,
    currents: np.ndarray,
    r: float,
    phi: float,
    z: float,
) -> tuple[float, float, float]:
    """(B_r, B_phi, B_z) at one point, for occasional use."""
    br, bp, bz = VacuumField(table, currents)(r, phi, z)
    return float(br[0]), float(bp[0]), float(bz[0])


def field_on_axis_scan(
    table: vmecpp.MagneticFieldResponseTable,
    currents: np.ndarray,
    r: float,
    num_phi: int = 64,
) -> np.ndarray:
    """|B| along the circle of radius ``r`` in the midplane, over one field period."""
    vacuum = VacuumField(table, currents)
    phi = np.linspace(0.0, vacuum.period, num_phi, endpoint=False)
    return vacuum.magnitude(np.full(num_phi, r), phi, np.zeros(num_phi))

# -- from full_torus --------------------------------------------------------------

def _expand_2d(coeff: np.ndarray | None, nfp: int, ntor_new: int) -> np.ndarray | None:
    """Move (m, n) coefficients to (m, n * nfp) on a wider toroidal index range."""
    if coeff is None:
        return None
    mpol, width = coeff.shape
    ntor_old = (width - 1) // 2
    out = np.zeros((mpol, 2 * ntor_new + 1))
    for n in range(-ntor_old, ntor_old + 1):
        target = n * nfp
        if abs(target) <= ntor_new:
            out[:, ntor_new + target] = coeff[:, ntor_old + n]
    return out


def _expand_axis(coeff: np.ndarray | None, nfp: int, ntor_new: int) -> np.ndarray | None:
    if coeff is None:
        return None
    out = np.zeros(ntor_new + 1)
    for n in range(min(len(coeff), ntor_new // nfp + 1)):
        out[n * nfp] = coeff[n]
    return out

# -- from harmonics ----------------------------------------------------------------

# Toroidal harmonic conventions: complex amplitudes, helical shifts, and interpolation.
#
# Three conventions that return a plausible wrong answer when they are broken, so they are
# carried in one place and guarded in ``tests/``.

#: Radius and sampling of the midplane circle the error-field harmonics are measured on.
HARMONIC_RADIUS_M = 6.2
HARMONIC_POINTS = 256


def radial_harmonics(
    vacuum, points: int = HARMONIC_POINTS, radius_m: float = HARMONIC_RADIUS_M
) -> np.ndarray:
    """Complex toroidal harmonics of the radial field on a midplane circle.

    Complex so callers difference the amplitudes before taking the modulus.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    b_r, _, _ = vacuum(np.full_like(phi, radius_m), phi, np.zeros_like(phi))
    return 2.0 * np.fft.rfft(b_r) / len(b_r)


def n1_amplitude(shift: np.ndarray) -> float:
    """n = 1 amplitude of an axis displacement, the larger of its two components, in metres."""
    shift = np.asarray(shift, dtype=float)
    radial = np.abs(np.fft.rfft(shift[:, 0]) / len(shift)) * 2.0
    vertical = np.abs(np.fft.rfft(shift[:, 1]) / len(shift)) * 2.0
    return float(max(radial[1], vertical[1]))


def normal_field_spectrum(
    vacuum, equilibrium, num_theta: int = 64, num_zeta: int = 160,
    reference_t: float | None = None,
) -> dict[tuple[int, int], complex]:
    """Normal-field harmonics on the boundary as ``{(m, n): amplitude}`` normalised to
    ``reference_t`` (else the field's own mean), n counted over the whole torus."""
    wout = equilibrium.wout
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
    rmnc, zmns = np.asarray(wout.rmnc)[:, -1], np.asarray(wout.zmns)[:, -1]

    theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, num_zeta, endpoint=False)
    theta2d, zeta2d = np.meshgrid(theta, zeta, indexing="ij")
    angle = (
        xm[None, None, :] * theta2d[:, :, None] - xn[None, None, :] * zeta2d[:, :, None]
    )
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    r = cos_a @ rmnc
    z = sin_a @ zmns
    dr_dtheta = -(sin_a * xm[None, None, :]) @ rmnc
    dz_dtheta = (cos_a * xm[None, None, :]) @ zmns
    dr_dzeta = (sin_a * xn[None, None, :]) @ rmnc
    dz_dzeta = -(cos_a * xn[None, None, :]) @ zmns

    # The surface normal in cylindrical components, as the cross product of the two
    # tangents. Its toroidal component carries the R of the toroidal tangent.
    normal_r = -r * dz_dtheta
    normal_phi = dz_dtheta * dr_dzeta - dr_dtheta * dz_dzeta
    normal_z = r * dr_dtheta
    magnitude = np.sqrt(normal_r**2 + normal_phi**2 + normal_z**2)

    b_r, b_phi, b_z = vacuum(r.ravel(), zeta2d.ravel(), z.ravel())
    projected = (
        b_r.reshape(r.shape) * normal_r
        + b_phi.reshape(r.shape) * normal_phi
        + b_z.reshape(r.shape) * normal_z
    ) / np.maximum(magnitude, 1e-30)
    reference = (
        float(reference_t)
        if reference_t is not None
        else float(np.mean(np.sqrt(b_r**2 + b_phi**2 + b_z**2)))
    )

    spectrum = np.fft.fft2(projected / max(reference, 1e-30)) / projected.size
    out: dict[tuple[int, int], complex] = {}
    for m in range(0, 8):
        for n in range(-10, 11):
            out[(m, n)] = complex(spectrum[m % num_theta, (-n) % num_zeta]) * 2.0
    return out


def refine(points: np.ndarray, samples: int = 4096) -> np.ndarray:
    """Trigonometric interpolation of a closed filament onto a dense curve by zero-padding."""
    points = np.asarray(points, dtype=float)
    spectrum = np.fft.rfft(points, axis=0)
    count = len(points)
    padded = np.zeros((samples // 2 + 1, points.shape[1]), dtype=complex)
    keep = min(len(spectrum), len(padded))
    padded[:keep] = spectrum[:keep]
    # An even point count carries a Nyquist term that has to be halved when it is spread
    # over the two conjugate slots the longer transform gives it.
    if count % 2 == 0 and keep == len(spectrum):
        padded[count // 2] *= 0.5
    return np.fft.irfft(padded, n=samples, axis=0) * (samples / count)
