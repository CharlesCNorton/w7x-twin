"""Plasma-facing geometry: the vessel contour and the components inside it."""

from __future__ import annotations

import dataclasses
import numpy as np
from pathlib import Path


# -- from vessel ------------------------------------------------------------------

@dataclasses.dataclass
class Vessel:
    """Closed poloidal contours of the plasma vessel, one per toroidal cut."""

    phi: np.ndarray  # (n_cuts,) radians, spanning one field period
    r: np.ndarray  # (n_cuts, n_poloidal) metres
    z: np.ndarray  # (n_cuts, n_poloidal) metres
    num_field_periods: int

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.num_field_periods

    def cut_at(self, phi: float) -> tuple[np.ndarray, np.ndarray]:
        """Vessel contour at one toroidal angle, linearly interpolated between cuts."""
        angle = np.mod(phi, self.period)
        index = np.searchsorted(self.phi, angle) - 1
        index = int(np.clip(index, 0, len(self.phi) - 2))
        span = self.phi[index + 1] - self.phi[index]
        weight = 0.0 if span == 0 else (angle - self.phi[index]) / span
        r = (1 - weight) * self.r[index] + weight * self.r[index + 1]
        z = (1 - weight) * self.z[index] + weight * self.z[index + 1]
        return r, z

    def resample(self, phi_grid: np.ndarray) -> ResampledVessel:
        """Precompute contours on a fixed toroidal grid."""
        cuts = [self.cut_at(float(p)) for p in phi_grid]
        return ResampledVessel(
            r=np.array([c[0] for c in cuts]),
            z=np.array([c[1] for c in cuts]),
            period=self.period,
            num_grid=len(phi_grid),
        )

    def bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.r.min()),
            float(self.r.max()),
            float(self.z.min()),
            float(self.z.max()),
        )


@dataclasses.dataclass
class ResampledVessel:
    """Vessel contours on the tracer's toroidal grid."""

    r: np.ndarray  # (num_grid, n_poloidal)
    z: np.ndarray
    period: float
    num_grid: int

    def outside(self, r: np.ndarray, z: np.ndarray, grid_index: int) -> np.ndarray:
        """True outside the slice's contour; non-finite points count outside."""
        inside = inside_contour(
            r, z, self.r[grid_index % self.num_grid], self.z[grid_index % self.num_grid]
        )
        return ~inside | ~np.isfinite(np.asarray(r)) | ~np.isfinite(np.asarray(z))


def inside_contour(
    point_r: np.ndarray, point_z: np.ndarray, contour_r: np.ndarray, contour_z: np.ndarray
) -> np.ndarray:
    """True for each point inside one closed (R, Z) contour, by crossing-number test."""
    z0 = contour_z[:, None]
    z1 = np.roll(contour_z, -1)[:, None]
    r0 = contour_r[:, None]
    r1 = np.roll(contour_r, -1)[:, None]

    point_r = np.asarray(point_r)[None, :]
    point_z = np.asarray(point_z)[None, :]

    straddles = (z0 > point_z) != (z1 > point_z)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing_r = r0 + (point_z - z0) * (r1 - r0) / (z1 - z0)
    hits = straddles & (point_r < crossing_r)
    return (np.count_nonzero(hits, axis=0) % 2) == 1


def load_vessel(path: str | Path, num_field_periods: int = 5) -> Vessel:
    """Read a ``vessel.part`` description: cuts of one field period, in cm and degrees."""
    tokens = Path(path).read_text().splitlines()
    header = tokens[1].split()
    num_cuts, num_poloidal, nfp = int(header[0]), int(header[1]), int(header[2])
    if nfp != num_field_periods:
        raise ValueError(f"vessel file declares nfp={nfp}, expected {num_field_periods}")

    values: list[float] = []
    for line in tokens[2:]:
        values.extend(float(v) for v in line.split())

    phi = np.empty(num_cuts)
    r = np.empty((num_cuts, num_poloidal))
    z = np.empty((num_cuts, num_poloidal))

    cursor = 0
    for cut in range(num_cuts):
        phi[cut] = np.deg2rad(values[cursor])
        cursor += 1
        for point in range(num_poloidal):
            # Each point is R, Z in centimetres followed by its index.
            r[cut, point] = values[cursor] / 100.0
            z[cut, point] = values[cursor + 1] / 100.0
            cursor += 3

    return Vessel(phi=phi, r=r, z=z, num_field_periods=num_field_periods)

# -- from components --------------------------------------------------------------

COMPONENT_FILES: dict[str, str] = {
    "divhorn9.t": "divertor horizontal target, upper",
    "divvern8.t": "divertor vertical target, upper",
    "divhoran7.b": "divertor horizontal target, lower",
    "bafhor1.t": "baffle, horizontal upper",
    "bafhor2.b": "baffle, horizontal lower",
    "bafhormid.t": "baffle, horizontal mid",
    "bafver1.t": "baffle, vertical upper",
    "bafver2.b": "baffle, vertical lower",
    "bafvern8.t": "baffle, vertical n8",
    "scraper_06_25_2013.t": "scraper element",
}

#: Components that receive the strike lines rather than merely intercepting them.
TARGET_COMPONENTS = (
    "divertor horizontal target, upper",
    "divertor vertical target, upper",
    "divertor horizontal target, lower",
)

#: The ten divertor units: one upper and one lower per module. IPP's component
#: database labels every plasma-facing element by module 1-5, by upper or lower unit,
#: and by section within the unit, and carries 1620 divertor, 3227 baffle and 4914
#: heat-shield components under that scheme. Five-fold replication of one unit is
#: therefore an approximation of the identity, not of the geometry: the contour is
#: shared, but a strike belongs to one named unit.
NUM_DIVERTOR_UNITS = 10


def unit_of(
    phi: np.ndarray, z: np.ndarray, num_field_periods: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """(module, is_upper) per strike from its toroidal angle and midplane side; unstruck is module 0."""
    period = 2.0 * np.pi / num_field_periods
    phi = np.asarray(phi, dtype=float)
    z = np.asarray(z, dtype=float)
    struck = np.isfinite(phi)
    module = np.floor(np.mod(np.where(struck, phi, 0.0), 2.0 * np.pi) / period)
    return np.where(struck, module + 1, 0).astype(int), z > 0.0


def arc_position(
    component: "Component", r: np.ndarray, z: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, float]:
    """Arc position of each strike along the component contour, with the contour's total length, in metres."""
    period = 2.0 * np.pi / NUM_FIELD_PERIODS_DEFAULT
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    phi = np.mod(np.asarray(phi, dtype=float), period)

    # Contours are stellarator symmetric, so a strike below the midplane is compared
    # against the mirrored contour.
    cut_index = np.argmin(
        np.abs(np.mod(component.phi, period)[None, :] - phi[:, None]), axis=1
    )
    lengths = np.hypot(np.diff(component.r, axis=1), np.diff(component.z, axis=1))
    arc = np.concatenate(
        [np.zeros((len(component.phi), 1)), np.cumsum(lengths, axis=1)], axis=1
    )

    out = np.empty(r.shape)
    for index in range(len(r)):
        cut = int(cut_index[index])
        contour_r = component.r[cut]
        contour_z = component.z[cut] if z[index] >= 0 else -component.z[cut]

        # Project onto each segment rather than snapping to a vertex: the vertices are
        # tens of millimetres apart, which is the scale the strike line moves over, so
        # snapping would quantise the answer away.
        start_r, start_z = contour_r[:-1], contour_z[:-1]
        span_r = contour_r[1:] - start_r
        span_z = contour_z[1:] - start_z
        squared = span_r**2 + span_z**2
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.clip(
                ((r[index] - start_r) * span_r + (z[index] - start_z) * span_z)
                / np.where(squared > 0, squared, 1.0),
                0.0,
                1.0,
            )
        distance = np.hypot(
            r[index] - (start_r + t * span_r), z[index] - (start_z + t * span_z)
        )
        segment = int(np.argmin(distance))
        out[index] = arc[cut, segment] + t[segment] * np.sqrt(squared[segment])
    return out, float(np.mean(arc[:, -1]))


#: Field period count the contours are given over, used when reducing a strike angle.
NUM_FIELD_PERIODS_DEFAULT = 5


def target_arc_frame(
    elements: list["Component"], names: tuple[str, ...] = TARGET_COMPONENTS
) -> dict[str, tuple[float, bool, float]]:
    """Join the divertor targets on one continuous arc coordinate: ``{name: (offset, reversed, length)}`` in metres."""
    ordered = [
        element
        for name in names
        for element in elements
        if element.name == name and not name.endswith("lower")
    ]
    if not ordered:
        return {}

    def ends(element: "Component") -> tuple[np.ndarray, np.ndarray]:
        cut = len(element.phi) // 2
        return (
            np.array([element.r[cut, 0], element.z[cut, 0]]),
            np.array([element.r[cut, -1], element.z[cut, -1]]),
        )

    def length(element: "Component") -> float:
        cut = len(element.phi) // 2
        return float(
            np.sum(np.hypot(np.diff(element.r[cut]), np.diff(element.z[cut])))
        )

    frame: dict[str, tuple[float, bool, float]] = {}
    offset = 0.0
    previous_end = None
    for element in ordered:
        start, finish = ends(element)
        span = length(element)
        reverse = False
        if previous_end is not None:
            # Join at whichever end lies nearer the previous target's far end.
            reverse = np.linalg.norm(finish - previous_end) < np.linalg.norm(
                start - previous_end
            )
        frame[element.name] = (offset, reverse, span)
        offset += span
        previous_end = start if reverse else finish
    return frame


def surface_frame(
    component: "Component", r: np.ndarray, z: np.ndarray, phi: np.ndarray,
    num_field_periods: int = NUM_FIELD_PERIODS_DEFAULT,
) -> dict[str, np.ndarray]:
    """Both surface derivatives at each strike, its element cut index, and its (across, along)
    element coordinates, interpolated between cuts rather than snapped."""
    period = 2.0 * np.pi / num_field_periods
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # The lower units are the (phi, Z) -> (-phi, -Z) image of the stored contours, so a
    # strike below the midplane is matched against the cut at the reflected angle.
    lower = z < 0.0
    angle = np.mod(np.where(lower, -phi, phi), period)
    stored = np.mod(component.phi, period)
    cut_index = np.argmin(np.abs(stored[None, :] - angle[:, None]), axis=1)

    num_cuts, num_poloidal = component.r.shape
    tangent_r = np.empty(r.shape)
    tangent_z = np.empty(r.shape)
    dr_dphi = np.empty(r.shape)
    dz_dphi = np.empty(r.shape)
    across = np.zeros(r.shape)
    along = np.zeros(r.shape)

    for index in range(r.size):
        cut = int(cut_index[index])
        mirror = -1.0 if lower[index] else 1.0

        # The cut pair the strike falls between, and where between them it is.
        neighbour = cut + (1 if angle[index] >= stored[cut] else -1)
        neighbour = int(np.clip(neighbour, 0, num_cuts - 1))
        low, high = (cut, neighbour) if neighbour > cut else (neighbour, cut)
        span = component.phi[high] - component.phi[low]
        weight = 0.0 if span == 0.0 else float((angle[index] - stored[low]) / span)
        weight = min(max(weight, 0.0), 1.0)
        across[index] = weight

        contour_r = (1.0 - weight) * component.r[low] + weight * component.r[high]
        contour_z = mirror * (
            (1.0 - weight) * component.z[low] + weight * component.z[high]
        )
        station = int(
            np.argmin(np.hypot(contour_r - r[index], contour_z - z[index]))
        )
        arc = np.concatenate(
            [[0.0], np.cumsum(np.hypot(np.diff(contour_r), np.diff(contour_z)))]
        )
        along[index] = float(arc[station])

        first = max(station - 1, 0)
        last = min(station + 1, num_poloidal - 1)
        tangent_r[index] = contour_r[last] - contour_r[first]
        tangent_z[index] = contour_z[last] - contour_z[first]

        if high == low or span == 0.0:
            dr_dphi[index] = 0.0
            dz_dphi[index] = 0.0
            continue
        # Under (phi, Z) -> (-phi, -Z) the two sign changes cancel on Z and not on R, so
        # the reflected instance reverses the toroidal derivative of R and keeps that of Z.
        dr_dphi[index] = (
            mirror * (component.r[high, station] - component.r[low, station]) / span
        )
        dz_dphi[index] = (
            component.z[high, station] - component.z[low, station]
        ) / span

    return {
        "tangent_r": tangent_r,
        "tangent_z": tangent_z,
        "dr_dphi": dr_dphi,
        "dz_dphi": dz_dphi,
        "element": cut_index,
        "across": across,
        "along": along,
    }


def base_name(name: str) -> str:
    """Element name without its upper or lower qualifier, which the unit supplies."""
    for suffix in (", upper", ", lower"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@dataclasses.dataclass
class Component:
    """One plasma-facing component as toroidal cuts of a poloidal contour."""

    name: str
    phi: np.ndarray  # (n_cuts,) radians
    r: np.ndarray  # (n_cuts, n_poloidal) metres
    z: np.ndarray

    def area_m2(self) -> float:
        """Surface area, from the poloidal arc length swept toroidally."""
        total = 0.0
        for cut in range(len(self.phi) - 1):
            arc = np.sum(
                np.hypot(np.diff(self.r[cut]), np.diff(self.z[cut]))
            )
            radius = float(np.mean(self.r[cut]))
            total += arc * radius * abs(self.phi[cut + 1] - self.phi[cut])
        return float(total)


def load_component(path: str | Path, name: str | None = None) -> Component:
    """Read a component file of toroidal cuts, the point stride inferred from the token count."""
    path = Path(path)
    tokens = path.read_text().split("\n")
    header = tokens[1].split()
    num_cuts, num_poloidal = int(header[0]), int(header[1])

    values: list[float] = []
    for line in tokens[2:]:
        values.extend(float(v) for v in line.split())

    stride = round((len(values) / num_cuts - 1) / num_poloidal)
    if stride not in (2, 3):
        raise ValueError(f"{path.name}: cannot infer point stride (got {stride})")

    phi = np.empty(num_cuts)
    r = np.empty((num_cuts, num_poloidal))
    z = np.empty((num_cuts, num_poloidal))

    cursor = 0
    for cut in range(num_cuts):
        phi[cut] = np.deg2rad(values[cursor])
        cursor += 1
        for point in range(num_poloidal):
            r[cut, point] = values[cursor] / 100.0
            z[cut, point] = values[cursor + 1] / 100.0
            cursor += stride

    return Component(name=name or path.name, phi=phi, r=r, z=z)


@dataclasses.dataclass
class ComponentSet:
    """Components resampled onto an integration grid, ready for intersection tests."""

    names: list[str]
    #: contours[slice] is a list of (component index, R array, Z array).
    contours: list[list[tuple[int, np.ndarray, np.ndarray]]]
    period: float

    @property
    def num_slices(self) -> int:
        return len(self.contours)

    def intersect(
        self,
        r0: np.ndarray,
        z0: np.ndarray,
        r1: np.ndarray,
        z1: np.ndarray,
        slice_index: int,
    ) -> np.ndarray:
        """Component index struck by each step segment by vectorised segment crossing, or -1."""
        hit = np.full(np.shape(r0), -1, dtype=int)
        entries = self.contours[slice_index % self.num_slices]
        if not entries:
            return hit

        dx = r1 - r0
        dy = z1 - z0
        for component, contour_r, contour_z in entries:
            ax = contour_r[:-1][:, None]
            ay = contour_z[:-1][:, None]
            bx = (contour_r[1:] - contour_r[:-1])[:, None]
            by = (contour_z[1:] - contour_z[:-1])[:, None]

            denominator = dx[None, :] * by - dy[None, :] * bx
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((ax - r0[None, :]) * by - (ay - z0[None, :]) * bx) / denominator
                u = ((ax - r0[None, :]) * dy - (ay - z0[None, :]) * dx) / denominator
            crossing = (
                np.isfinite(t) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
            )
            struck = np.any(crossing, axis=0)
            hit = np.where((hit < 0) & struck, component, hit)
        return hit


def load_components(
    directory: str | Path, files: dict[str, str] | None = None
) -> list[Component]:
    directory = Path(directory)
    files = files or COMPONENT_FILES
    out = []
    for filename, name in files.items():
        path = directory / filename
        if path.exists():
            out.append(load_component(path, name))
    return out


def build_component_set(
    components: list[Component],
    phi_grid: np.ndarray,
    num_field_periods: int = 5,
    stellarator_symmetric: bool = True,
) -> ComponentSet:
    """Components on a one-period toroidal grid, replicated and reflected through (phi, Z) -> (-phi, -Z)."""
    period = 2.0 * np.pi / num_field_periods
    names = [c.name for c in components]
    contours: list[list[tuple[int, np.ndarray, np.ndarray]]] = [
        [] for _ in phi_grid
    ]
    # Half a grid spacing: a cut is assigned to the nearest slice.
    spacing = period / len(phi_grid)

    for index, component in enumerate(components):
        instances = [(component.phi, component.r, component.z)]
        if stellarator_symmetric:
            instances.append((-component.phi, component.r, -component.z))

        for phi_values, r_values, z_values in instances:
            for cut, angle in enumerate(phi_values):
                slot = int(np.round(np.mod(angle, period) / spacing)) % len(phi_grid)
                contours[slot].append((index, r_values[cut], z_values[cut]))

    return ComponentSet(names=names, contours=contours, period=period)
