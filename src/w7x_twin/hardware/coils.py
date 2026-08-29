"""Auxiliary coil sets, the finite conductor build, and deflection under load."""

from __future__ import annotations

import dataclasses
import numpy as np
from pathlib import Path
from w7x_twin.hardware.machine import NUM_FIELD_PERIODS
from w7x_twin.hardware.machine import _COILS_START, load_coils
from w7x_twin.hardware.walls import Vessel


# -- from auxiliary_coils ---------------------------------------------------------

#: Trim-band outboard outer-vessel radius from the released CAD, inside the 7.000-7.690 m error-field pin.
OUTER_VESSEL_RADIUS_M = 7.570

#: The type B coil sits closer in than the type A coils.
TYPE_B_RADIUS_M = 7.3

TRIM_A_WIDTH_M, TRIM_A_HEIGHT_M, TRIM_A_TURNS = 3.5, 3.3, 48
TRIM_B_WIDTH_M, TRIM_B_HEIGHT_M, TRIM_B_TURNS = 2.8, 2.2, 72

CONTROL_LENGTH_M = 2.05
CONTROL_WIDTH_M = 0.35
CONTROL_TURNS = 8

#: Poloidal fraction of the vessel contour the divertor units sit at, symmetric about the midplane.
CONTROL_POLOIDAL_FRACTION = 0.22


@dataclasses.dataclass
class ConstructedGroup:
    """One circuit's worth of constructed filaments."""

    key: str
    name: str
    turns: int
    filaments: list[np.ndarray]


# -- measured geometry, when the coil database is reachable -------------------------

#: REST face of IPP's coil database; the host resolves only inside the institute.
COILSDB_REST = "http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest"

#: CoilsDB record identifiers per the FusionSC listing; the type B trim coil is identified from geometry.
MAIN_COIL_IDS = tuple(range(160, 230))
TRIM_COIL_IDS = (350, 241, 351, 352, 535)
CONTROL_COIL_IDS = tuple(range(230, 240))


def fetch_coilsdb_filaments(
    coil_ids, base_url: str = COILSDB_REST, timeout: float = 30.0
) -> list[np.ndarray]:
    """One (n, 3) Cartesian polyline per requested CoilsDB record; raises when unreachable."""
    import json
    import urllib.error
    import urllib.request

    out: list[np.ndarray] = []
    for coil_id in coil_ids:
        url = f"{base_url.rstrip('/')}/coil/{int(coil_id)}/data"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RuntimeError(
                f"coil database unreachable at {url}: {error}. The service resolves "
                "only inside IPP; without it the trim and control coil geometry "
                "remains the reconstruction in this module."
            ) from error
        vertices = payload["polylineFilament"]["vertices"]
        out.append(
            np.stack(
                [
                    np.asarray(vertices["x1"], dtype=float),
                    np.asarray(vertices["x2"], dtype=float),
                    np.asarray(vertices["x3"], dtype=float),
                ],
                axis=-1,
            )
        )
    return out


def measured_auxiliary_coils(
    base_url: str = COILSDB_REST, timeout: float = 30.0
) -> list[ConstructedGroup]:
    """Trim and control coils as measured filaments, a drop-in for the constructed groups."""
    trim = fetch_coilsdb_filaments(TRIM_COIL_IDS, base_url, timeout)
    control = fetch_coilsdb_filaments(CONTROL_COIL_IDS, base_url, timeout)

    # The type B coil is the one shaped around a space restriction, 2.8 x 2.2 m
    # against 3.5 x 3.3 m, so it is the smallest of the five by winding perimeter.
    # Its record number does not distinguish it.
    perimeters = [
        float(np.sum(np.linalg.norm(np.diff(filament, axis=0), axis=1)))
        for filament in trim
    ]
    type_b = int(np.argmin(perimeters))

    groups: list[ConstructedGroup] = []
    type_a_count = 0
    for index, filament in enumerate(trim):
        if index == type_b:
            key, turns = "trim_b1", TRIM_B_TURNS
        else:
            type_a_count += 1
            key, turns = f"trim_a{type_a_count}", TRIM_A_TURNS
        groups.append(
            ConstructedGroup(key=key, name=key.upper(), turns=turns, filaments=[filament])
        )

    groups.extend(
        ConstructedGroup(f"cc{m + 1}u", f"CONTROL_{m + 1}U", CONTROL_TURNS, [path])
        for m, path in enumerate(control[0::2])
    )
    groups.extend(
        ConstructedGroup(f"cc{m + 1}l", f"CONTROL_{m + 1}L", CONTROL_TURNS, [path])
        for m, path in enumerate(control[1::2])
    )
    return groups


def _rounded_rectangle(
    width: float, height: float, corner: float, num_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """A closed rectangle with rounded corners, in its own plane."""
    half_w, half_h = width / 2.0, height / 2.0
    corner = min(corner, half_w * 0.9, half_h * 0.9)
    angle = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    # Superellipse-style rounding keeps the perimeter smooth without corner cases.
    exponent = max(2.0, min(width, height) / max(corner, 1e-6))
    u = np.cos(angle)
    v = np.sin(angle)
    scale = (np.abs(u) ** exponent + np.abs(v) ** exponent) ** (-1.0 / exponent)
    return half_w * u * scale, half_h * v * scale


def trim_coils(
    num_points: int = 96,
    radius: float = OUTER_VESSEL_RADIUS_M,
    type_b_radius: float = TYPE_B_RADIUS_M,
    width_scale: float = 1.0,
    height_scale: float = 1.0,
    corner: float = 0.4,
    toroidal_offset: float = 0.0,
    vertical_offset: float = 0.0,
) -> list[ConstructedGroup]:
    """Five trim coils as planar rectangles wound as their published packs, one filament
    per turn; the mounting radius, rounding and placement inferences stay parameters."""
    groups: list[ConstructedGroup] = []
    mounting = radius
    for module in range(NUM_FIELD_PERIODS):
        is_type_b = module == NUM_FIELD_PERIODS - 1
        width = (TRIM_B_WIDTH_M if is_type_b else TRIM_A_WIDTH_M) * width_scale
        height = (TRIM_B_HEIGHT_M if is_type_b else TRIM_A_HEIGHT_M) * height_scale
        pack = TRIM_B_PACK if is_type_b else TRIM_A_PACK
        radius_here = type_b_radius if is_type_b else mounting

        # Middle of the module, displaced by the offsets under test.
        phi = (module + 0.5) * 2.0 * np.pi / NUM_FIELD_PERIODS + toroidal_offset
        local_w, local_z = _rounded_rectangle(width, height, corner, num_points)
        local_z = local_z + vertical_offset
        radius = radius_here

        # The plane is normal to the radial direction: local width runs toroidally.
        toroidal = np.array([-np.sin(phi), np.cos(phi), 0.0])
        centre = np.array([radius * np.cos(phi), radius * np.sin(phi), 0.0])
        points = (
            centre[None, :]
            + local_w[:, None] * toroidal[None, :]
            + local_z[:, None] * np.array([0.0, 0.0, 1.0])[None, :]
        )
        points = np.vstack([points, points[:1]])

        key = "trim_b1" if is_type_b else f"trim_a{module + 1}"
        groups.append(
            ConstructedGroup(
                key=key, name=key.upper(), turns=1.0,
                filaments=expand_coil(points, pack),
            )
        )
    return groups


def control_coils(vessel: Vessel, num_points: int = 80) -> list[ConstructedGroup]:
    """Ten control coils as vessel-wall saddles spanning the published 2.05 m by 0.35 m."""
    upper: list[np.ndarray] = []
    lower: list[np.ndarray] = []

    for module in range(NUM_FIELD_PERIODS):
        phi_centre = (module + 0.5) * 2.0 * np.pi / NUM_FIELD_PERIODS
        reference_r, _ = vessel.cut_at(phi_centre)
        radius = float(np.mean(reference_r))
        half_phi = 0.5 * CONTROL_LENGTH_M / radius

        for sign, bucket in ((+1.0, upper), (-1.0, lower)):
            bucket.append(
                _vessel_saddle(vessel, phi_centre, half_phi, sign, num_points)
            )

    out: list[ConstructedGroup] = []
    for module, path in enumerate(upper, start=1):
        out.append(
            ConstructedGroup(
                f"cc{module}u", f"CONTROL_{module}U", 1.0,
                expand_coil(path, CONTROL_PACK),
            )
        )
    for module, path in enumerate(lower, start=1):
        out.append(
            ConstructedGroup(
                f"cc{module}l", f"CONTROL_{module}L", 1.0,
                expand_coil(path, CONTROL_PACK),
            )
        )
    return out


def _vessel_saddle(
    vessel: Vessel,
    phi_centre: float,
    half_phi: float,
    sign: float,
    num_points: int,
) -> np.ndarray:
    """A rectangular saddle on the vessel wall, in (toroidal, poloidal) extent."""
    num_poloidal = vessel.r.shape[1]
    centre_index = int(CONTROL_POLOIDAL_FRACTION * num_poloidal)
    if sign < 0:
        centre_index = num_poloidal - centre_index

    def wall_point(phi: float, index: float) -> np.ndarray:
        # The poloidal position is interpolated between contour vertices rather than
        # snapped to one: the vertices are some 80 mm apart and the coil spans 350 mm,
        # so snapping quantises its short sides onto five of them and repeats points,
        # and a repeated point has a zero tangent and no winding frame.
        r, z = vessel.cut_at(phi)
        low = int(np.floor(index)) % num_poloidal
        high = (low + 1) % num_poloidal
        weight = float(index) - np.floor(index)
        radius = (1.0 - weight) * r[low] + weight * r[high]
        height = (1.0 - weight) * z[low] + weight * z[high]
        return np.array([radius * np.cos(phi), radius * np.sin(phi), height])

    # Poloidal half-width, converted from metres to contour indices.
    r_cut, z_cut = vessel.cut_at(phi_centre)
    spacing = float(
        np.mean(np.hypot(np.diff(r_cut), np.diff(z_cut)))
    )
    half_index = max(1, int(round(0.5 * CONTROL_WIDTH_M / spacing)))

    per_side = max(4, num_points // 4)
    phis = np.linspace(phi_centre - half_phi, phi_centre + half_phi, per_side)
    indices = np.linspace(
        centre_index - half_index, centre_index + half_index, per_side
    )

    # Each side starts where the previous one ended, and the fourth returns to the
    # first side's start, so every side but the first drops its opening vertex and
    # the last drops its closing one. Carried whole, the walk repeats all four
    # corners and closes twice, which leaves the winding one frame short of its own
    # points when the pack is expanded along it.
    path = [wall_point(p, centre_index - half_index) for p in phis]
    path += [wall_point(phis[-1], i) for i in indices[1:]]
    path += [wall_point(p, centre_index + half_index) for p in phis[::-1][1:]]
    path += [wall_point(phis[0], i) for i in indices[::-1][1:-1]]
    points = np.array(path)
    return np.vstack([points, points[:1]])


def write_extended_coils_file(
    base_coils_path: str,
    output_path: str,
    vessel: Vessel,
    include_trim: bool = True,
    include_control: bool = True,
) -> list[str]:
    """Write a MAKEGRID coils file with the added coils at one filament per turn, current column one."""
    from pathlib import Path

    from w7x_twin.hardware.machine import _COILS_START, _W7X_GROUP_TO_CIRCUIT, load_coils

    base = load_coils(base_coils_path)
    lines = Path(base_coils_path).read_text().splitlines()
    # The header comments quote the marker string, so match the marker itself.
    start = next(i for i, l in enumerate(lines) if _COILS_START in l)

    header = lines[: start + 1]
    body = lines[start + 1 :]
    # Keep everything up to the terminating "end" of the original filament list.
    end_index = next(i for i, l in enumerate(body) if l.strip().lower() == "end")
    kept = body[:end_index]

    extra: list[ConstructedGroup] = []
    if include_trim:
        extra.extend(trim_coils())
    if include_control:
        extra.extend(control_coils(vessel))

    group_index = base.num_circuits
    out: list[str] = []
    for group in extra:
        group_index += 1
        for filament in group.filaments:
            for point in filament[:-1]:
                out.append(
                    f"{point[0]: .10E} {point[1]: .10E} {point[2]: .10E} "
                    f"{float(group.turns): .10E}"
                )
            closing = filament[-1]
            out.append(
                f"{closing[0]: .10E} {closing[1]: .10E} {closing[2]: .10E} "
                f"{0.0: .10E} {group_index} {group.name}"
            )

    Path(output_path).write_text(
        "\n".join(header + kept + out + ["end", ""])
    )
    return [_W7X_GROUP_TO_CIRCUIT[n] for n in base.group_names] + [
        g.key for g in extra
    ]

# -- from finite_build ------------------------------------------------------------

CONDUCTOR_SIZE_M = 0.016
#: Turn-to-turn pitch: conductor plus the half-overlapped glass tape and layer mats.
TURN_PITCH_M = 0.0175


@dataclasses.dataclass(frozen=True)
class WindingPack:
    """Turn layout of one coil type as (layers, turns per layer), each direction with its own pitch."""

    layers: int
    turns_per_layer: int
    pitch: float = TURN_PITCH_M
    turn_pitch: float | None = None

    @property
    def pitch_across_turns(self) -> float:
        return self.pitch if self.turn_pitch is None else self.turn_pitch

    @property
    def turns(self) -> int:
        return self.layers * self.turns_per_layer


#: 108 turns as twelve layers of nine; both pitches measured from the released WP_model.stp brick faces.
NON_PLANAR_PACK = WindingPack(
    layers=12, turns_per_layer=9,
    pitch=(0.21588 - CONDUCTOR_SIZE_M) / 11,
    turn_pitch=TURN_PITCH_M,
)
#: 36 planar-coil turns as six by six, both pitches from the released model's brick faces.
PLANAR_PACK = WindingPack(
    layers=6, turns_per_layer=6,
    pitch=(0.10894 - CONDUCTOR_SIZE_M) / 5,
    turn_pitch=(0.1050 - CONDUCTOR_SIZE_M) / 5,
)

#: Trim conductor 16.26 mm square, insulation 0.4 mm turn and 1.0 mm layer, eight pancakes of six or nine (Rummel et al., IEEE Trans. Appl. Supercond. 22 (2012) 4201704).
TRIM_CONDUCTOR_M = 0.01626
TRIM_A_PACK = WindingPack(
    layers=8, turns_per_layer=6,
    pitch=TRIM_CONDUCTOR_M + 0.001,
    turn_pitch=TRIM_CONDUCTOR_M + 0.0004,
)
TRIM_B_PACK = WindingPack(
    layers=8, turns_per_layer=9,
    pitch=TRIM_CONDUCTOR_M + 0.001,
    turn_pitch=TRIM_CONDUCTOR_M + 0.0004,
)
#: Control coils: 16 mm conductor, eight published turns, the two-by-four split inferred.
CONTROL_PACK = WindingPack(layers=2, turns_per_layer=4, pitch=0.017)


def constructed_digest() -> str:
    """Digest of the constructed-coil constants: their part of the geometry version."""
    import hashlib

    values = (
        OUTER_VESSEL_RADIUS_M, TYPE_B_RADIUS_M,
        TRIM_A_WIDTH_M, TRIM_A_HEIGHT_M, TRIM_A_TURNS,
        TRIM_B_WIDTH_M, TRIM_B_HEIGHT_M, TRIM_B_TURNS,
        TRIM_CONDUCTOR_M,
        TRIM_A_PACK.layers, TRIM_A_PACK.turns_per_layer,
        TRIM_A_PACK.pitch, TRIM_A_PACK.pitch_across_turns,
        TRIM_B_PACK.layers, TRIM_B_PACK.turns_per_layer,
        CONTROL_LENGTH_M, CONTROL_WIDTH_M, CONTROL_TURNS,
        CONTROL_PACK.layers, CONTROL_PACK.turns_per_layer, CONTROL_PACK.pitch,
        CONTROL_POLOIDAL_FRACTION,
    )
    text = ",".join(f"{value!r}" for value in values)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-section frame along a curve, parallel-transported from the best-fit plane normal.

    The caller owns the closure and hands over an open path, so stripping one here
    as well would return one frame fewer than there are points on a winding whose
    open path happens to end where it began.
    """
    closed = points
    tangent = np.gradient(closed, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]

    centred = closed - closed.mean(axis=0)
    # Smallest singular direction of the point cloud is the best-fit plane normal.
    normal_seed = np.linalg.svd(centred, full_matrices=False)[2][-1]

    normal = np.empty_like(tangent)
    current = normal_seed - np.dot(normal_seed, tangent[0]) * tangent[0]
    current /= np.linalg.norm(current)
    normal[0] = current
    for i in range(1, len(tangent)):
        # Parallel transport: remove the component along the new tangent, renormalise.
        current = current - np.dot(current, tangent[i]) * tangent[i]
        norm = np.linalg.norm(current)
        current = normal[i - 1] if norm < 1e-12 else current / norm
        normal[i] = current

    binormal = np.cross(tangent, normal)
    binormal /= np.linalg.norm(binormal, axis=1)[:, None]
    return normal, binormal


def expand_coil(points: np.ndarray, pack: WindingPack) -> list[np.ndarray]:
    """One filament per turn, laid out on the winding-pack cross-section."""
    closed = points[:-1] if np.allclose(points[0], points[-1]) else points
    normal, binormal = _frame(closed)

    layer_offsets = (np.arange(pack.layers) - (pack.layers - 1) / 2.0) * pack.pitch
    turn_offsets = (
        np.arange(pack.turns_per_layer) - (pack.turns_per_layer - 1) / 2.0
    ) * pack.pitch_across_turns

    filaments = []
    for du in layer_offsets:
        for dv in turn_offsets:
            path = closed + du * normal + dv * binormal
            filaments.append(np.vstack([path, path[:1]]))
    return filaments


def write_finite_build_coils_file(
    base_coils_path: str | Path, output_path: str | Path
) -> dict[str, int]:
    """Write a coils file with every pack expanded to one filament per turn, current column one."""
    base = load_coils(base_coils_path)
    lines = Path(base_coils_path).read_text().splitlines()
    # The header comments quote the marker string, so match the marker itself.
    start = next(i for i, l in enumerate(lines) if _COILS_START in l)
    header = lines[: start + 1]
    # "periods", "begin filament" and "mirror NIL" must survive into the new file.
    preamble = lines[start + 1 : start + 4]

    counts: dict[str, int] = {}
    out: list[str] = []
    for group_index, (key, name, coils) in enumerate(
        zip(base.circuit_keys, base.group_names, base.filaments, strict=True), start=1
    ):
        pack = NON_PLANAR_PACK if base.turns()[group_index - 1] == 108 else PLANAR_PACK
        total = 0
        for coil in coils:
            for filament in expand_coil(coil, pack):
                for point in filament[:-1]:
                    out.append(
                        f"{point[0]: .10E} {point[1]: .10E} {point[2]: .10E} "
                        f"{1.0: .10E}"
                    )
                closing = filament[-1]
                out.append(
                    f"{closing[0]: .10E} {closing[1]: .10E} {closing[2]: .10E} "
                    f"{0.0: .10E} {group_index} {name}"
                )
                total += 1
        counts[key] = total

    Path(output_path).write_text("\n".join(header + preamble + out + ["end", ""]))
    return counts

# -- from coil_deformation --------------------------------------------------------

MU0 = 4.0e-7 * np.pi

#: Young's modulus of the cast steel casing that carries the load, in pascals.
CASING_MODULUS_PA = 1.9e11

#: Cross-section of a non-planar winding pack: the turn direction from the released
#: model's brick faces, the layer direction the distance between one brick's face planes.
NON_PLANAR_WIDTH_M = 0.1557
NON_PLANAR_HEIGHT_M = 0.2159
#: The planar packs, from the same model.
PLANAR_WIDTH_M = 0.1050
PLANAR_HEIGHT_M = 0.1089
#: Wall thickness of the casing around the pack. The casing outer section is the pack plus
#: twice this, and the second moment is the difference of the two rectangles.
CASING_WALL_M = 0.05

#: Unsupported span between coil supports, in metres; the deflection scales as its fourth power.
SUPPORT_SPAN_M = 1.5


@dataclasses.dataclass(frozen=True)
class Deflection:
    """What one coil does under load."""

    circuit: str
    coil: int
    force_per_metre_n_m: np.ndarray
    total_force_n: float
    peak_deflection_m: float
    mean_deflection_m: float
    displaced: np.ndarray


def second_moment(width_m: float, height_m: float, wall_m: float = CASING_WALL_M) -> float:
    """Smaller principal second moment of the hollow casing about its weaker axis, in m^4."""
    outer_w, outer_h = width_m + 2.0 * wall_m, height_m + 2.0 * wall_m
    about_height = (outer_w * outer_h**3 - width_m * height_m**3) / 12.0
    about_width = (outer_h * outer_w**3 - height_m * width_m**3) / 12.0
    return min(about_height, about_width)


def segment_field(
    filaments: list[np.ndarray],
    currents: np.ndarray,
    points: np.ndarray,
    exclude: int | None = None,
) -> np.ndarray:
    """Biot-Savart field at ``points`` from every filament but the excluded self term."""
    field = np.zeros_like(points)
    for index, (path, current) in enumerate(zip(filaments, currents, strict=True)):
        if exclude is not None and index == exclude:
            continue
        start = path[:-1]
        end = path[1:]
        segment = end - start
        middle = 0.5 * (start + end)
        for chunk in range(0, len(points), 256):
            block = points[chunk : chunk + 256]
            offset = block[:, None, :] - middle[None, :, :]
            distance = np.linalg.norm(offset, axis=2)
            np.maximum(distance, 1e-6, out=distance)
            cross = np.cross(segment[None, :, :], offset)
            field[chunk : chunk + 256] += (
                MU0 * current / (4.0 * np.pi)
            ) * np.sum(cross / distance[:, :, None] ** 3, axis=1)
    return field


def deflect(
    filaments: list[np.ndarray],
    currents: np.ndarray,
    index: int,
    circuit: str,
    modulus_pa: float = CASING_MODULUS_PA,
    width_m: float = NON_PLANAR_WIDTH_M,
    height_m: float = NON_PLANAR_HEIGHT_M,
    span_m: float = SUPPORT_SPAN_M,
) -> Deflection:
    """Coil deflection as a doubly clamped beam, w L^4 / (384 E I) per span under the imposed Lorentz load."""
    path = np.asarray(filaments[index], dtype=float)
    start, end = path[:-1], path[1:]
    segment = end - start
    length = np.linalg.norm(segment, axis=1)
    middle = 0.5 * (start + end)

    field = segment_field(filaments, currents, middle, exclude=index)
    # Force per unit length: I t x B, with t the unit tangent.
    tangent = segment / np.maximum(length[:, None], 1e-12)
    force_per_metre = currents[index] * np.cross(tangent, field)

    # The component that bends the pack is the one across it, which is everything but the
    # part along the winding.
    along = np.sum(force_per_metre * tangent, axis=1)
    transverse = force_per_metre - along[:, None] * tangent
    magnitude = np.linalg.norm(transverse, axis=1)

    perimeter = float(length.sum())
    inertia = second_moment(width_m, height_m)
    load = float(np.average(magnitude, weights=length))
    peak = load * span_m**4 / (384.0 * modulus_pa * inertia)

    # The displaced winding: each span bows along its own local transverse force and is
    # pinned at the supports, so the pattern repeats once per span around the perimeter.
    fraction = np.concatenate([[0.0], np.cumsum(length)])
    spans = max(1.0, perimeter / span_m)
    bow = np.sin(np.pi * spans * fraction / perimeter) ** 2
    direction = np.zeros_like(path)
    direction[:-1] = transverse / np.maximum(magnitude[:, None], 1e-30)
    direction[-1] = direction[0]
    scale = peak * (
        np.concatenate([magnitude, magnitude[-1:]]) / max(magnitude.max(), 1e-30)
    )
    displaced = path + direction * (bow * scale)[:, None]

    return Deflection(
        circuit=circuit,
        coil=index,
        force_per_metre_n_m=magnitude,
        total_force_n=float(np.sum(magnitude * length)),
        peak_deflection_m=float(peak),
        mean_deflection_m=float(np.mean(np.linalg.norm(displaced - path, axis=1))),
        displaced=displaced,
    )
