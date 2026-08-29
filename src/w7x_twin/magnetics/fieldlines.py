"""Vacuum field-line tracing, Poincare sections, and strike records."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from w7x_twin.hardware.walls import (
    Component,
    base_name as component_base_name,
    build_component_set,
    unit_of as components_unit_of,
)
from w7x_twin.magnetics.field import VacuumField
from w7x_twin.hardware.walls import Vessel


@dataclasses.dataclass
class Poincare:
    """Intersections of traced field lines with one toroidal plane."""

    r: np.ndarray
    z: np.ndarray
    line_index: np.ndarray
    plane_phi: float
    turns_completed: np.ndarray
    strikes: "Strikes | None" = None
    #: (n_samples, n_lines, 3) cylindrical (R, phi, Z) along each line, when the
    #: trace was asked to record it. Points after a line stops are NaN.
    path: np.ndarray | None = None

    def by_line(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (self.r[self.line_index == i], self.z[self.line_index == i])
            for i in range(int(self.line_index.max()) + 1)
        ]


@dataclasses.dataclass
class Strikes:
    """Where traced field lines met the wall, and how far they ran to get there."""

    struck: np.ndarray  # bool per line
    r: np.ndarray
    z: np.ndarray
    phi: np.ndarray
    connection_length_m: np.ndarray
    start_r: np.ndarray
    #: Index into ``component_names`` of the element struck, or -1 for the bare wall.
    component: np.ndarray | None = None
    component_names: list[str] | None = None

    def tally(self) -> dict[str, int]:
        """Number of lines terminating on each named element."""
        if self.component is None or self.component_names is None:
            return {}
        counts: dict[str, int] = {}
        for index, name in enumerate(self.component_names):
            hits = int(np.count_nonzero(self.struck & (self.component == index)))
            if hits:
                counts[name] = hits
        wall = int(np.count_nonzero(self.struck & (self.component < 0)))
        if wall:
            counts["vessel wall"] = wall
        return counts

    def unit_tally(self, num_field_periods: int = 5) -> dict[tuple[str, int, str], int]:
        """Line counts keyed by (element, module, unit), the five-fold periodicity check."""
        if self.component is None or self.component_names is None:
            return {}
        module, is_upper = components_unit_of(self.phi, self.z, num_field_periods)
        counts: dict[tuple[str, int, str], int] = {}
        for line in np.flatnonzero(self.struck):
            index = int(self.component[line])
            name = (
                component_base_name(self.component_names[index])
                if index >= 0
                else "vessel wall"
            )
            key = (name, int(module[line]), "upper" if is_upper[line] else "lower")
            counts[key] = counts.get(key, 0) + 1
        return counts

    @classmethod
    def concatenate(cls, parts: "list[Strikes]") -> "Strikes":
        """One record over several fans; the component names are the first fan's."""
        return cls(
            struck=np.concatenate([p.struck for p in parts]),
            r=np.concatenate([p.r for p in parts]),
            z=np.concatenate([p.z for p in parts]),
            phi=np.concatenate([p.phi for p in parts]),
            connection_length_m=np.concatenate([p.connection_length_m for p in parts]),
            start_r=np.concatenate([p.start_r for p in parts]),
            component=np.concatenate([p.component for p in parts]),
            component_names=parts[0].component_names,
        )


def _derivatives(
    vacuum: VacuumField, r: np.ndarray, z: np.ndarray, phi: float
) -> tuple[np.ndarray, np.ndarray]:
    """dR/dphi and dZ/dphi along a field line."""
    b_r, b_phi, b_z = vacuum(r, phi, z)
    # B_phi does not vanish anywhere inside a stellarator vessel, so no guard is
    # needed beyond the NaN that leaves the grid.
    return r * b_r / b_phi, r * b_z / b_phi


def _arc_rate(
    vacuum: VacuumField, r: np.ndarray, z: np.ndarray, phi: float
) -> np.ndarray:
    """dl/dphi, the arc length a field line covers per radian of toroidal angle."""
    b_r, b_phi, b_z = vacuum(r, phi, z)
    return r * np.sqrt(b_r * b_r + b_phi * b_phi + b_z * b_z) / np.abs(b_phi)


def trace(
    vacuum: VacuumField,
    r_start: np.ndarray,
    z_start: np.ndarray,
    turns: int = 200,
    steps_per_period: int = 120,
    plane_phi: float = 0.0,
    winding_reference: int | None = None,
    vessel: Vessel | None = None,
    wall_check_every: int = 1,
    components: list[Component] | None = None,
    record_path_every: int | None = None,
    sense: int = +1,
) -> tuple[Poincare, np.ndarray | None]:
    """RK4-trace field lines and record their ``plane_phi`` crossings, winding about the
    co-traced axis, strikes against ``vessel``, and optionally each line's sampled path."""
    r = np.array(r_start, dtype=float)
    z = np.array(z_start, dtype=float)
    num_lines = r.size
    sense = 1 if sense >= 0 else -1

    # One "turn" is a full toroidal revolution; the field repeats every period.
    steps_per_turn = steps_per_period * vacuum.num_field_periods
    dphi = sense * 2.0 * np.pi / steps_per_turn

    # Resample the wall onto the integration grid so the containment test needs no
    # interpolation inside the loop. The grid always runs in the positive direction and
    # a reversed trace indexes it backwards, since the geometry is what it is.
    grid_phi = plane_phi + np.arange(steps_per_period) * abs(dphi)
    wall = vessel.resample(grid_phi) if vessel is not None else None
    elements = (
        build_component_set(components, grid_phi, vacuum.num_field_periods)
        if components
        else None
    )

    hits_r: list[np.ndarray] = []
    hits_z: list[np.ndarray] = []
    hits_line: list[np.ndarray] = []
    hits_turn: list[np.ndarray] = []
    alive = np.ones(num_lines, dtype=bool)
    index = np.arange(num_lines)

    winding = None
    if winding_reference is not None:
        winding = np.zeros(num_lines)
        previous_angle = np.arctan2(
            z - z[winding_reference], r - r[winding_reference]
        )

    path: list[np.ndarray] = []
    step_counter = 0

    arc_length = np.zeros(num_lines)
    struck = np.zeros(num_lines, dtype=bool)
    strike_r = np.full(num_lines, np.nan)
    strike_z = np.full(num_lines, np.nan)
    strike_phi = np.full(num_lines, np.nan)
    strike_component = np.full(num_lines, -1, dtype=int)
    start_r = r.copy()
    # Position at the previous surface check, so the span between checks is tested
    # rather than only the last step of it.
    check_r, check_z = r.copy(), z.copy()

    for turn in range(turns):
        phi = plane_phi + sense * turn * 2.0 * np.pi
        # Record the crossing at the start of each revolution.
        live = alive & np.isfinite(r) & np.isfinite(z)
        hits_r.append(r[live].copy())
        hits_z.append(z[live].copy())
        hits_line.append(index[live].copy())
        hits_turn.append(np.full(int(live.sum()), turn))

        for step in range(steps_per_turn):
            p = phi + step * dphi
            if record_path_every and step_counter % record_path_every == 0:
                path.append(np.stack([r, np.full(num_lines, p), z], axis=-1))
            step_counter += 1
            k1r, k1z = _derivatives(vacuum, r, z, p)
            k2r, k2z = _derivatives(
                vacuum, r + 0.5 * dphi * k1r, z + 0.5 * dphi * k1z, p + 0.5 * dphi
            )
            k3r, k3z = _derivatives(
                vacuum, r + 0.5 * dphi * k2r, z + 0.5 * dphi * k2z, p + 0.5 * dphi
            )
            k4r, k4z = _derivatives(
                vacuum, r + dphi * k3r, z + dphi * k3z, p + dphi
            )
            if wall is not None:
                # Arc length is a distance, so the reversed trace accumulates the same
                # positive amount per step as the forward one; signed, the two branches of a
                # connection length cancel instead of adding.
                arc_length = arc_length + np.where(
                    struck, 0.0, np.nan_to_num(_arc_rate(vacuum, r, z, p) * abs(dphi))
                )

            r = r + (dphi / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
            z = z + (dphi / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)

            if wall is not None and step % wall_check_every == 0:
                # Plasma-facing components sit inside the vessel, so they are tested
                # first and the bare wall catches whatever passes between them.
                slot = sense * step
                element = (
                    elements.intersect(check_r, check_z, r, z, slot)
                    if elements is not None
                    else np.full(num_lines, -1, dtype=int)
                )
                hit = wall.outside(r, z, slot) | (element >= 0)
                fresh = hit & ~struck
                if fresh.any():
                    strike_r[fresh] = r[fresh]
                    strike_z[fresh] = z[fresh]
                    strike_phi[fresh] = np.mod(p + dphi, 2.0 * np.pi)
                    strike_component[fresh] = element[fresh]
                    struck |= fresh
                # A line that has reached a surface stops being followed.
                r = np.where(struck, np.nan, r)
                z = np.where(struck, np.nan, z)
                check_r, check_z = r.copy(), z.copy()

            if winding is not None:
                angle = np.arctan2(
                    z - z[winding_reference], r - r[winding_reference]
                )
                delta = angle - previous_angle
                # Each step turns the poloidal angle by far less than pi, so the
                # branch cut is the only source of a large jump.
                delta = np.where(delta > np.pi, delta - 2 * np.pi, delta)
                delta = np.where(delta < -np.pi, delta + 2 * np.pi, delta)
                winding = winding + np.where(np.isfinite(delta), delta, 0.0)
                previous_angle = np.where(np.isfinite(angle), angle, previous_angle)

        alive &= np.isfinite(r) & np.isfinite(z)
        if not alive.any():
            break

    section = Poincare(
        r=np.concatenate(hits_r),
        z=np.concatenate(hits_z),
        line_index=np.concatenate(hits_line),
        plane_phi=plane_phi,
        turns_completed=np.concatenate(hits_turn),
        strikes=Strikes(
            struck=struck,
            r=strike_r,
            z=strike_z,
            phi=strike_phi,
            connection_length_m=np.where(struck, arc_length, np.nan),
            start_r=start_r,
            component=strike_component,
            component_names=elements.names if elements is not None else None,
        )
        if wall is not None
        else None,
        path=np.stack(path) if path else None,
    )
    if winding is None:
        return section, None
    completed = max(1, turn if not alive.all() else turns)
    return section, np.abs(winding) / (2.0 * np.pi * completed)


@dataclasses.dataclass
class Connection:
    """A line followed to a surface both ways; ``length_m`` is a lower bound where ``closed`` is False."""

    forward: Strikes
    reverse: Strikes
    length_m: np.ndarray
    closed: np.ndarray

    @property
    def struck(self) -> np.ndarray:
        """Lines that reached a surface either way."""
        return self.forward.struck | self.reverse.struck


def connection_lengths(
    vacuum: VacuumField,
    r_start: np.ndarray,
    z_start: np.ndarray,
    vessel: Vessel,
    components: list[Component] | None = None,
    turns: int = 200,
    plane_phi: float = 0.0,
    **keywords,
) -> Connection:
    """Trace each line both ways and sum the two arcs: the connection length."""
    both = [
        trace(
            vacuum, r_start, z_start, turns=turns, plane_phi=plane_phi,
            vessel=vessel, components=components, sense=sense, **keywords,
        )[0].strikes
        for sense in (+1, -1)
    ]
    forward, reverse = both
    lengths = np.nansum(
        np.stack([forward.connection_length_m, reverse.connection_length_m]), axis=0
    )
    closed = forward.struck & reverse.struck
    return Connection(
        forward=forward,
        reverse=reverse,
        length_m=np.where(forward.struck | reverse.struck, lengths, np.nan),
        closed=closed,
    )


#: Axes already located this process, keyed on the field digest and the search inputs.
_AXIS_MEMO: dict[str, tuple[float, float]] = {}


def find_axis(
    vacuum: VacuumField,
    r_guess: float = 5.93,
    z_guess: float = 0.0,
    plane_phi: float = 0.0,
    iterations: int = 40,
    tolerance: float = 1e-8,
    steps_per_period: int = 120,
    cache_dir: str | Path | None = "cache",
) -> tuple[float, float]:
    """Magnetic axis as the return map's fixed point, by damped finite-difference Newton.

    The axis is a pure function of the field, so it is memoised on the field digest, in
    memory and under ``cache_dir``; pass ``cache_dir=None`` to search unconditionally.
    """
    key = (
        f"{vacuum.digest()}_{plane_phi!r}_{r_guess!r}_{z_guess!r}"
        f"_{iterations}_{tolerance!r}_{steps_per_period}"
    )
    if key in _AXIS_MEMO:
        return _AXIS_MEMO[key]
    path = Path(cache_dir) / "axes.json" if cache_dir is not None else None
    stored: dict[str, list[float]] = {}
    if path is not None and path.exists():
        stored = json.loads(path.read_text())
        if key in stored:
            r, z = float(stored[key][0]), float(stored[key][1])
            _AXIS_MEMO[key] = (r, z)
            return r, z

    r, z = float(r_guess), float(z_guess)
    epsilon = 1e-4

    def one_turn(r0: np.ndarray, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result, _ = trace(
            vacuum, r0, z0, turns=2, steps_per_period=steps_per_period,
            plane_phi=plane_phi,
        )
        last = result.turns_completed == 1
        return result.r[last], result.z[last]

    for _ in range(iterations):
        probe_r = np.array([r, r + epsilon, r])
        probe_z = np.array([z, z, z + epsilon])
        mapped_r, mapped_z = one_turn(probe_r, probe_z)
        if mapped_r.size != 3:
            raise RuntimeError("field line left the grid while locating the axis")

        f = np.array([mapped_r[0] - r, mapped_z[0] - z])
        if np.linalg.norm(f) < tolerance:
            break
        jacobian = np.array(
            [
                [(mapped_r[1] - mapped_r[0]) / epsilon - 1.0,
                 (mapped_r[2] - mapped_r[0]) / epsilon],
                [(mapped_z[1] - mapped_z[0]) / epsilon,
                 (mapped_z[2] - mapped_z[0]) / epsilon - 1.0],
            ]
        )
        try:
            step = np.linalg.solve(jacobian, -f)
        except np.linalg.LinAlgError:
            break
        r, z = r + 0.5 * step[0], z + 0.5 * step[1]

    _AXIS_MEMO[key] = (r, z)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stored[key] = [r, z]
        staging = path.with_suffix(".json.tmp")
        staging.write_text(json.dumps(stored, indent=0))
        staging.replace(path)
    return r, z


def fan_starts(
    vacuum: VacuumField,
    wout,
    span: tuple[float, float],
    count: int,
    plane_phi: float = 0.0,
) -> tuple[np.ndarray, float, float, float]:
    """Launch radii across ``span`` in boundary-normalised units at one plane, with the
    traced axis and the boundary's outboard radius they are measured against."""
    from w7x_twin.mhd import diagnostics

    r_axis, z_axis = find_axis(vacuum, plane_phi=plane_phi)
    r_cut, _ = diagnostics.boundary_cut(wout, plane_phi)
    outboard = float(r_cut.max())
    starts = r_axis + np.linspace(*span, count) * (outboard - r_axis)
    return starts, r_axis, z_axis, outboard


def midplane_island_span(
    section: Poincare, r_axis: float, z_axis: float, min_points: int = 4
) -> tuple[float, int]:
    """Largest per-line radial extent of section points at the outboard midplane, in
    metres, with the number of lines measured; NaN where no line leaves enough points."""
    outboard = (np.abs(section.z - z_axis) < 0.02) & (section.r > r_axis)
    spans = []
    for line in np.unique(section.line_index[outboard]):
        here = outboard & (section.line_index == line)
        if int(here.sum()) >= min_points:
            spans.append(float(section.r[here].max() - section.r[here].min()))
    if not spans:
        return float("nan"), 0
    return float(max(spans)), len(spans)
