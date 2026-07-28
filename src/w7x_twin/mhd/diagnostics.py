"""An equilibrium reduced to scalars: geometry, field, transform, and stability."""

from __future__ import annotations

import dataclasses
import numpy as np
import vmecpp


# -- from diagnostics -------------------------------------------------------------

MU0 = 4.0e-7 * np.pi

#: Low-order rationals the W7-X island divertor is built around. The toroidal mode
#: number is a multiple of the field period count, so these are the transforms at
#: which a chain of five islands sits in the boundary region.
DIVERTOR_RESONANCES: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 5),
    (5, 4),
)


@dataclasses.dataclass
class Diagnostics:
    """Scalar summary of one equilibrium."""

    # Geometry
    major_radius_m: float
    minor_radius_m: float
    aspect_ratio: float
    plasma_volume_m3: float

    # Field
    b_axis_t: float
    b_bean_t: float
    b_triangle_t: float
    b_min_t: float
    b_max_t: float
    mirror_ratio: float
    mirror_percent: float

    # Transform
    iota_axis: float
    iota_edge: float
    iota_min: float
    iota_max: float
    resonances_crossed: tuple[str, ...]

    # Pressure and stability
    beta_total: float
    beta_poloidal: float
    beta_toroidal: float
    beta_axis: float
    average_pressure_pa: float
    stored_energy_j: float
    net_toroidal_current_a: float
    magnetic_well_depth: float
    mercier_unstable_fraction: float

    # Shape response
    axis_shift_m: float
    #: The axis against the boundary's own centre, which is the frame a published
    #: Shafranov shift describes: the boundary also moves at finite beta in a
    #: free-boundary solve, and that motion is not the internal shift.
    axis_shift_in_boundary_m: float

    def as_dict(self) -> dict[str, float | tuple[str, ...]]:
        return dataclasses.asdict(self)


def _axis_position(wout: vmecpp.VmecWOut, phi: float = 0.0) -> tuple[float, float]:
    """(R, Z) of the magnetic axis at toroidal angle ``phi``."""
    xn = np.asarray(wout.xn)
    xm = np.asarray(wout.xm)
    on_axis = xm == 0
    angle = -xn[on_axis] * phi
    r = float(np.sum(np.asarray(wout.rmnc)[on_axis, 0] * np.cos(angle)))
    z = float(np.sum(np.asarray(wout.zmns)[on_axis, 0] * np.sin(angle)))
    return r, z


def _internal_axis_offset(wout: vmecpp.VmecWOut, num_phi: int = 24) -> float:
    """Axis radius less the boundary's m = 0 radius, averaged over a field period."""
    xn = np.asarray(wout.xn)
    xm = np.asarray(wout.xm)
    on_centre = xm == 0
    rmnc = np.asarray(wout.rmnc)
    offsets = []
    for phi in np.linspace(0.0, 2.0 * np.pi / wout.nfp, num_phi, endpoint=False):
        angle = -xn[on_centre] * phi
        axis_r = float(np.sum(rmnc[on_centre, 0] * np.cos(angle)))
        centre_r = float(np.sum(rmnc[on_centre, -1] * np.cos(angle)))
        offsets.append(axis_r - centre_r)
    return float(np.mean(offsets))


def field_on_axis(wout: vmecpp.VmecWOut, num_phi: int = 256) -> np.ndarray:
    """|B| along the axis over one period, extrapolated from the two innermost half-grid surfaces."""
    xn_nyq = np.asarray(wout.xn_nyq)
    xm_nyq = np.asarray(wout.xm_nyq)
    bmnc = np.asarray(wout.bmnc)
    on_axis = xm_nyq == 0

    # bmnc lives on the half grid; index 0 is unused, 1 and 2 are the two innermost.
    inner = 1.5 * bmnc[on_axis, 1] - 0.5 * bmnc[on_axis, 2]
    n_values = xn_nyq[on_axis]

    phi = np.linspace(0.0, 2.0 * np.pi / wout.nfp, num_phi, endpoint=False)
    return np.array(
        [float(np.sum(inner * np.cos(-n_values * p))) for p in phi]
    )


def specific_volume(wout: vmecpp.VmecWOut) -> tuple[np.ndarray, np.ndarray]:
    """``dV/ds`` against ``s`` on the half grid, with the unused first entry dropped."""
    vp = np.abs(np.asarray(wout.vp))[1:]
    ns = int(wout.ns)
    s = (np.arange(1, ns) - 0.5) / (ns - 1)
    return s, vp


def magnetic_well_depth(wout: vmecpp.VmecWOut) -> float:
    """Fractional decrease of dV/ds from axis to edge: positive is a magnetic well."""
    _, vp = specific_volume(wout)
    return float((vp[0] - vp[-1]) / vp[0])


def crossed_resonances(iota: np.ndarray) -> tuple[str, ...]:
    lo, hi = float(np.min(iota)), float(np.max(iota))
    out = []
    for n, m in DIVERTOR_RESONANCES:
        value = n / m
        if lo - 1e-9 <= value <= hi + 1e-9:
            out.append(f"{n}/{m}")
    return tuple(out)


def analyse(
    output: vmecpp.VmecOutput,
    vacuum_reference: vmecpp.VmecOutput | None = None,
) -> Diagnostics:
    """Reduce one equilibrium to its scalar machine quantities."""
    wout = output.wout
    iota = np.asarray(wout.iotaf)
    b_axis_profile = field_on_axis(wout)

    axis_r, _ = _axis_position(wout)
    shift = 0.0
    internal = 0.0
    if vacuum_reference is not None:
        reference_r, _ = _axis_position(vacuum_reference.wout)
        shift = axis_r - reference_r
        internal = _internal_axis_offset(wout) - _internal_axis_offset(
            vacuum_reference.wout
        )

    # The Mercier criterion is driven by the pressure gradient, so at zero pressure
    # its sign is numerical noise and reporting a fraction would be meaningless.
    has_pressure = float(wout.betatotal) > 0.0
    mercier = np.asarray(output.mercier.DMerc)
    interior = mercier[1:-1]
    unstable = (
        float(np.count_nonzero(interior < 0.0) / interior.size)
        if has_pressure and interior.size
        else float("nan")
    )

    b_max = float(np.max(b_axis_profile))
    b_min = float(np.min(b_axis_profile))
    # phi = 0 is the bean-shaped cross section, the half period the triangular one.
    b_bean = float(b_axis_profile[0])
    b_triangle = float(b_axis_profile[len(b_axis_profile) // 2])

    return Diagnostics(
        major_radius_m=float(wout.Rmajor_p),
        minor_radius_m=float(wout.Aminor_p),
        aspect_ratio=float(wout.aspect),
        plasma_volume_m3=float(wout.volume_p),
        b_axis_t=float(wout.b0),
        b_bean_t=b_bean,
        b_triangle_t=b_triangle,
        b_min_t=b_min,
        b_max_t=b_max,
        mirror_ratio=b_max / b_min,
        mirror_percent=100.0 * (b_max - b_min) / (b_max + b_min),
        iota_axis=float(iota[0]),
        iota_edge=float(iota[-1]),
        iota_min=float(np.min(iota)),
        iota_max=float(np.max(iota)),
        resonances_crossed=crossed_resonances(iota),
        beta_total=float(wout.betatotal),
        beta_poloidal=float(wout.betapol),
        beta_toroidal=float(wout.betator),
        beta_axis=float(wout.betaxis),
        average_pressure_pa=float(output.threed1_volumetrics.avg_p),
        # VMEC's int_ekin is 3/2 times the pressure integral, i.e. the stored
        # thermal energy of an isotropic plasma.
        stored_energy_j=float(output.threed1_volumetrics.int_ekin),
        net_toroidal_current_a=float(wout.ctor),
        magnetic_well_depth=magnetic_well_depth(wout),
        mercier_unstable_fraction=unstable,
        axis_shift_m=shift,
        axis_shift_in_boundary_m=internal,
    )


def flux_surface(
    wout: vmecpp.VmecWOut,
    surface: int,
    phi: float,
    num_theta: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """(R, Z) of one flux surface in the plane at toroidal angle ``phi``."""
    xn = np.asarray(wout.xn)
    xm = np.asarray(wout.xm)
    rmnc = np.asarray(wout.rmnc)[:, surface]
    zmns = np.asarray(wout.zmns)[:, surface]

    theta = np.linspace(0.0, 2.0 * np.pi, num_theta)
    angle = np.outer(theta, xm) - phi * xn[None, :]
    r = np.cos(angle) @ rmnc
    z = np.sin(angle) @ zmns
    return r, z


def boundary_cut(
    wout: vmecpp.VmecWOut, phi: float, num_theta: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """(R, Z) of the outermost flux surface at toroidal angle ``phi``."""
    return flux_surface(wout, int(wout.ns) - 1, phi, num_theta)


def cross_section(
    wout: vmecpp.VmecWOut, phi: float, num_surfaces: int = 8, num_theta: int = 256
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A nested set of flux-surface outlines at one toroidal angle."""
    ns = int(wout.ns)
    indices = np.unique(np.linspace(1, ns - 1, num_surfaces).astype(int))
    return [flux_surface(wout, int(i), phi, num_theta) for i in indices]


def iota_profile(wout: vmecpp.VmecWOut) -> tuple[np.ndarray, np.ndarray]:
    iota = np.asarray(wout.iotaf)
    s = np.linspace(0.0, 1.0, len(iota))
    return s, iota

# -- from stability ---------------------------------------------------------------

VACUUM_PERMEABILITY = 4.0e-7 * np.pi


@dataclasses.dataclass
class BallooningResult:
    """Local ballooning drive against the shear that stabilises it, per surface."""

    s: np.ndarray
    #: Normalised pressure gradient driving the mode.
    alpha: np.ndarray
    #: Magnetic shear s dq/ds / q, which stabilises it.
    shear: np.ndarray
    #: Critical alpha the shear supports.
    alpha_critical: np.ndarray
    unstable: np.ndarray

    @property
    def unstable_fraction(self) -> float:
        return float(np.count_nonzero(self.unstable) / max(self.unstable.size, 1))


@dataclasses.dataclass
class TearingResult:
    """Tearing index at one resonant surface."""

    #: Poloidal and toroidal mode numbers of the resonance.
    m: int
    n: int
    s_resonant: float
    #: Jump in the logarithmic derivative of the perturbed flux, per metre.
    delta_prime_per_m: float
    #: Positive means the resonance is tearing unstable at finite resistivity.
    unstable: bool


def ballooning(
    s: np.ndarray,
    pressure_pa: np.ndarray,
    iota: np.ndarray,
    field_t: float,
    minor_radius_m: float,
    major_radius_m: float,
) -> BallooningResult:
    """Ballooning drive alpha = -(2 mu_0 R q^2 / B^2) dp/dr against alpha_crit = 0.6 (1 + shear)."""
    s = np.asarray(s, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    transform = np.asarray(iota, dtype=float)
    radius = minor_radius_m * np.sqrt(np.clip(s, 1e-12, 1.0))
    safety = 1.0 / np.where(np.abs(transform) > 1e-9, transform, np.nan)

    gradient = np.gradient(pressure, radius)
    alpha = -(
        2.0 * VACUUM_PERMEABILITY * major_radius_m * safety**2 / field_t**2
    ) * gradient

    shear = radius / safety * np.gradient(safety, radius)
    critical = 0.6 * (1.0 + np.abs(shear))
    return BallooningResult(
        s=s,
        alpha=alpha,
        shear=shear,
        alpha_critical=critical,
        unstable=np.asarray(alpha > critical),
    )


def tearing_index(
    s: np.ndarray,
    current_density_a_m2: np.ndarray,
    iota: np.ndarray,
    m: int,
    n: int,
    minor_radius_m: float,
) -> TearingResult:
    """Newcomb delta-prime at the n/m resonance; positive means the resonance opens."""
    s = np.asarray(s, dtype=float)
    transform = np.asarray(iota, dtype=float)
    current = np.asarray(current_density_a_m2, dtype=float)
    radius = minor_radius_m * np.sqrt(np.clip(s, 1e-12, 1.0))

    resonant_iota = float(n) / float(m)
    crossings = np.where(np.diff(np.sign(transform - resonant_iota)))[0]
    if crossings.size == 0:
        return TearingResult(
            m=m, n=n, s_resonant=float("nan"),
            delta_prime_per_m=float("nan"), unstable=False,
        )
    index = int(crossings[0])
    s_resonant = float(
        np.interp(resonant_iota, transform[index : index + 2], s[index : index + 2])
    )
    r_s = minor_radius_m * np.sqrt(max(s_resonant, 1e-12))

    # Newcomb equation for psi: (r psi')' - (m^2/r) psi - (r J'/ (B (iota - iota_s))) psi = 0.
    # Integrated as a two-point shooting problem with the resonance excluded.
    def integrate(order: np.ndarray) -> float:
        psi, slope = 1.0e-6, 1.0e-6
        previous = radius[order[0]]
        for position in order[1:]:
            step = radius[position] - previous
            if step == 0.0:
                continue
            r = max(abs(radius[position]), 1e-6)
            distance = transform[position] - resonant_iota
            if abs(distance) < 1e-4:
                break
            drive = np.gradient(current, radius)[position] / (r * distance)
            curvature = (m**2 / r**2) * psi + drive * psi
            slope += curvature * step
            psi += slope * step
            previous = radius[position]
        return float(slope / psi) if psi != 0.0 else float("nan")

    inner = np.arange(0, index + 1)
    outer = np.arange(len(radius) - 1, index, -1)
    from_axis = integrate(inner)
    from_edge = integrate(outer)
    delta = (from_edge - from_axis) * r_s
    return TearingResult(
        m=m, n=n, s_resonant=s_resonant,
        delta_prime_per_m=float(delta),
        unstable=bool(np.isfinite(delta) and delta > 0.0),
    )


def resonances_in(
    iota: np.ndarray, max_m: int = 6
) -> list[tuple[int, int]]:
    """Low-order rationals n/m the transform profile crosses, in lowest terms."""
    from math import gcd

    transform = np.asarray(iota, dtype=float)
    low, high = float(np.min(transform)), float(np.max(transform))
    out = []
    for m in range(1, max_m + 1):
        for n in range(1, max_m + 1):
            if gcd(m, n) != 1:
                continue
            if low <= n / m <= high:
                out.append((m, n))
    return sorted(set(out))

@dataclasses.dataclass
class GlobalMode:
    """One global ideal interchange mode, from a radial eigenvalue problem."""

    m: int
    n: int
    #: Eigenvalue of the reduced energy functional. Negative is unstable.
    eigenvalue: float
    #: Surface the eigenfunction peaks on.
    peak_s: float
    unstable: bool


def global_modes(
    s: np.ndarray,
    pressure_pa: np.ndarray,
    iota: np.ndarray,
    magnetic_well: np.ndarray,
    minor_radius_m: float,
    major_radius_m: float,
    field_t: float,
    modes: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (3, 3), (1, 0), (2, 1)),
) -> list[GlobalMode]:
    """Global interchange eigenvalues of W[xi] = int f (dxi/dr)^2 + g xi^2 dr; negative lowest means unstable."""
    s = np.asarray(s, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    transform = np.asarray(iota, dtype=float)
    well = np.asarray(magnetic_well, dtype=float)
    if not np.all(np.isfinite(well)):
        raise ValueError(
            "the magnetic well profile carries a non-finite value; VMEC stores dV/ds with a "
            "zero at the axis, so normalising by it gives an infinite drive"
        )
    radius = np.maximum(minor_radius_m * np.sqrt(np.clip(s, 0.0, 1.0)), 1e-6)

    gradient = np.gradient(pressure, radius)
    # Pressure gradient against the well, in inverse metres, with the package's sign
    # convention: a positive well stabilises, so it raises the eigenvalue, and a hill lowers
    # it. The gradient is negative inside a confined plasma, hence the leading sign.
    drive = -2.0 * VACUUM_PERMEABILITY * gradient * well / field_t**2

    out: list[GlobalMode] = []
    for m, n in modes:
        if m == 0:
            continue
        # Parallel wavenumber of the mode, in inverse metres. It vanishes on the resonant
        # surface, which is where field-line bending stops opposing the mode.
        wavenumber = m * transform / radius - n / max(major_radius_m, 1e-9)
        # Bending carries metres so that f / dr^2 and the drive share inverse metres, which
        # is what lets the pressure enter the eigenvalue at all.
        bending = radius**3 * wavenumber**2

        # Finite-volume assembly on a grid whose spacing varies fifteenfold from axis to
        # edge: the bending is taken on the cell faces and each difference divided by its own
        # neighbour spacing. A single spacing for both neighbours is not a discretisation of
        # this operator and produces negative eigenvalues where the bending alone cannot.
        interior = np.arange(1, len(radius) - 1)
        if interior.size < 3:
            continue
        face = 0.5 * (bending[:-1] + bending[1:])
        count = interior.size
        main = np.zeros(count)
        lower = np.zeros(count - 1)
        for k, i in enumerate(interior):
            left_h = radius[i] - radius[i - 1]
            right_h = radius[i + 1] - radius[i]
            cell = 0.5 * (left_h + right_h)
            left = face[i - 1] / left_h
            right = face[i] / right_h
            main[k] = (left + right) / cell + drive[i]
            if k < count - 1:
                lower[k] = -right / cell
        # Symmetrise by the cell measure, which turns the operator self-adjoint so the
        # bending contributes a positive definite part and eigh applies.
        cells = np.array([
            0.5 * (radius[i + 1] - radius[i - 1]) for i in interior
        ])
        scale = np.sqrt(np.maximum(cells * np.maximum(radius[interior], 1e-9), 1e-30))
        operator = np.diag(main / np.maximum(radius[interior], 1e-9))
        for k in range(count - 1):
            coupling = lower[k] / np.sqrt(
                max(radius[interior][k], 1e-9) * max(radius[interior][k + 1], 1e-9)
            )
            operator[k, k + 1] = coupling
            operator[k + 1, k] = coupling
        values, vectors = np.linalg.eigh(operator)
        order = int(np.argmin(values))
        out.append(
            GlobalMode(
                m=m, n=n,
                eigenvalue=float(values[order]),
                peak_s=float(s[interior][int(np.argmax(np.abs(vectors[:, order])))]),
                unstable=bool(values[order] < 0.0),
            )
        )
    return out
