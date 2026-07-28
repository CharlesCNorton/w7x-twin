"""Reading the released CAD: STEP entities, solids, and tube cross-sections, in millimetres."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np

#: Entity types the walker keeps; all others are skipped at parse time.
KEPT = (
    "CARTESIAN_POINT",
    "VERTEX_POINT",
    "MANIFOLD_SOLID_BREP",
    "BREP_WITH_VOIDS",
    "CLOSED_SHELL",
    "OPEN_SHELL",
    "SHELL_BASED_SURFACE_MODEL",
    "ADVANCED_FACE",
    "FACE_OUTER_BOUND",
    "FACE_BOUND",
    "EDGE_LOOP",
    "ORIENTED_EDGE",
    "EDGE_CURVE",
    "VERTEX_LOOP",
    "PLANE",
)

_ENTITY = re.compile(r"^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\)\s*;\s*$")
_REFERENCE = re.compile(r"#(\d+)")
_TRIPLE = re.compile(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)")


def parse_entities(path: str | Path) -> dict[int, tuple[str, str]]:
    """Entity id to (type, argument text), for the kept types only."""
    out: dict[int, tuple[str, str]] = {}
    with open(path, "r", encoding="latin-1", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                continue
            match = _ENTITY.match(line.strip())
            if match is None:
                continue
            kind = match.group(2)
            if kind in KEPT:
                out[int(match.group(1))] = (kind, match.group(3))
    return out


def cartesian_point(argument: str) -> np.ndarray | None:
    triple = _TRIPLE.search(argument)
    if triple is None:
        return None
    return np.array([float(triple.group(k)) for k in (1, 2, 3)])


@dataclasses.dataclass
class Solid:
    """One B-rep solid: the exact corner vertices, and every point it references."""

    identifier: int
    vertices: np.ndarray
    points: np.ndarray


def solids(path: str | Path) -> list[Solid]:
    """Every solid of a STEP file, its points assigned by entity-reference closure from the solid root."""
    entities = parse_entities(path)
    roots = [
        identifier
        for identifier, (kind, _) in entities.items()
        if kind in ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS", "SHELL_BASED_SURFACE_MODEL")
    ]
    out = []
    for root in roots:
        seen: set[int] = set()
        stack = [root]
        vertices, points = [], []
        while stack:
            identifier = stack.pop()
            if identifier in seen or identifier not in entities:
                continue
            seen.add(identifier)
            kind, argument = entities[identifier]
            if kind == "CARTESIAN_POINT":
                value = cartesian_point(argument)
                if value is not None:
                    points.append(value)
                continue
            if kind == "VERTEX_POINT":
                for reference in _REFERENCE.findall(argument):
                    target = entities.get(int(reference))
                    if target and target[0] == "CARTESIAN_POINT":
                        value = cartesian_point(target[1])
                        if value is not None:
                            vertices.append(value)
                continue
            stack.extend(int(r) for r in _REFERENCE.findall(argument))
        if vertices or points:
            vertex_array = np.array(vertices) if vertices else np.empty((0, 3))
            out.append(
                Solid(
                    identifier=root,
                    vertices=vertex_array,
                    points=np.array(points) if points else vertex_array,
                )
            )
    return out


def _ear_clip(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangles of a simple planar polygon by ear clipping, robust to either winding."""
    count = len(polygon)
    if count < 3:
        return []
    indices = list(range(count))
    x, y = polygon[:, 0], polygon[:, 1]
    area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    sign = 1.0 if area >= 0.0 else -1.0
    out: list[tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3 and guard < 4 * count:
        guard += 1
        clipped = False
        for k in range(len(indices)):
            a, b, c = (
                indices[k - 1],
                indices[k],
                indices[(k + 1) % len(indices)],
            )
            u = polygon[b] - polygon[a]
            v = polygon[c] - polygon[b]
            if sign * (u[0] * v[1] - u[1] * v[0]) <= 1e-12:
                continue
            triangle = polygon[[a, b, c]]
            others = [i for i in indices if i not in (a, b, c)]
            if others and _any_inside(triangle, polygon[others]):
                continue
            out.append((a, b, c))
            indices.pop(k)
            clipped = True
            break
        if not clipped:
            break
    if len(indices) == 3:
        out.append((indices[0], indices[1], indices[2]))
    return out


def _any_inside(triangle: np.ndarray, points: np.ndarray) -> bool:
    """True if any point lies strictly inside the triangle."""
    a, b, c = triangle
    v0, v1 = c - a, b - a
    v2 = points - a
    dot00, dot01, dot11 = v0 @ v0, v0 @ v1, v1 @ v1
    dot02 = v2 @ v0
    dot12 = v2 @ v1
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(denominator) < 1e-18:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denominator
    v = (dot00 * dot12 - dot01 * dot02) / denominator
    return bool(np.any((u > 1e-9) & (v > 1e-9) & (u + v < 1.0 - 1e-9)))


def planar_face_triangles(
    path: str | Path, include_curved: bool = True
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Every face of a STEP file triangulated from its own edge loops.
    Curved edges are taken by chord and curved faces by the best-fit plane; counts tallies the approximation."""
    entities = parse_entities(path)

    def loop_vertices(loop_id: int) -> list[int]:
        kind, argument = entities.get(loop_id, ("", ""))
        if kind != "EDGE_LOOP":
            return []
        ordered: list[int] = []
        for edge_id in (int(r) for r in _REFERENCE.findall(argument)):
            edge_kind, edge_argument = entities.get(edge_id, ("", ""))
            if edge_kind != "ORIENTED_EDGE":
                continue
            references = [int(r) for r in _REFERENCE.findall(edge_argument)]
            forward = ".T." in edge_argument
            for curve_id in references:
                curve_kind, curve_argument = entities.get(curve_id, ("", ""))
                if curve_kind != "EDGE_CURVE":
                    continue
                ends = [int(r) for r in _REFERENCE.findall(curve_argument)[:2]]
                if len(ends) == 2:
                    ordered.append(ends[0] if forward else ends[1])
                break
        return ordered

    def vertex_point(vertex_id: int) -> np.ndarray | None:
        kind, argument = entities.get(vertex_id, ("", ""))
        if kind != "VERTEX_POINT":
            return None
        for reference in _REFERENCE.findall(argument):
            target = entities.get(int(reference))
            if target and target[0] == "CARTESIAN_POINT":
                return cartesian_point(target[1])
        return None

    positions: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    index_of: dict[int, int] = {}
    counts = {"planar": 0, "chorded": 0, "curved": 0, "degenerate": 0}

    for identifier, (kind, argument) in entities.items():
        if kind != "ADVANCED_FACE":
            continue
        references = [int(r) for r in _REFERENCE.findall(argument)]
        surface = next(
            (r for r in references if entities.get(r, ("",))[0] == "PLANE"), None
        )
        if surface is None and not include_curved:
            counts["curved"] += 1
            continue
        bounds = [
            r
            for r in references
            if entities.get(r, ("",))[0] in ("FACE_OUTER_BOUND", "FACE_BOUND")
        ]
        outer = None
        for bound in bounds:
            if entities[bound][0] == "FACE_OUTER_BOUND" or outer is None:
                loop = next(
                    (
                        int(r)
                        for r in _REFERENCE.findall(entities[bound][1])
                        if entities.get(int(r), ("",))[0] == "EDGE_LOOP"
                    ),
                    None,
                )
                if loop is not None:
                    outer = loop
                    if entities[bound][0] == "FACE_OUTER_BOUND":
                        break
        if outer is None:
            counts["degenerate"] += 1
            continue

        vertex_ids = loop_vertices(outer)
        points = [vertex_point(v) for v in vertex_ids]
        points = [p for p in points if p is not None]
        if len(points) < 3:
            counts["degenerate"] += 1
            continue
        polygon = np.array(points)
        centred = polygon - polygon.mean(axis=0)
        basis = np.linalg.svd(centred, full_matrices=False)[2][:2]
        local = centred @ basis.T
        ears = _ear_clip(local)
        if not ears:
            counts["degenerate"] += 1
            continue
        counts["planar" if surface is not None else "chorded"] += 1
        base_indices = []
        for vertex_id, point in zip(vertex_ids, points, strict=False):
            if vertex_id not in index_of:
                index_of[vertex_id] = len(positions)
                positions.append(point)
            base_indices.append(index_of[vertex_id])
        for a, b, c in ears:
            triangles.append((base_indices[a], base_indices[b], base_indices[c]))

    vertices = (np.array(positions) / 1e3).astype(np.float32)
    return vertices, np.array(triangles, dtype=np.uint32), counts


def all_points(path: str | Path) -> np.ndarray:
    """Every cartesian point of a file, when the solid structure is not needed."""
    out = []
    with open(path, "r", encoding="latin-1", errors="replace") as handle:
        for line in handle:
            if "CARTESIAN_POINT" not in line:
                continue
            value = _TRIPLE.search(line, line.find("CARTESIAN_POINT"))
            if value is not None:
                out.append([float(value.group(k)) for k in (1, 2, 3)])
    return np.array(out)


def minimal_rectangle(planar: np.ndarray) -> tuple[float, float]:
    """Width and height of the smallest-area enclosing rectangle of planar points, by rotating calipers."""
    if len(planar) < 3:
        return float("nan"), float("nan")
    hull = _convex_hull(planar)
    best = (float("inf"), float("nan"), float("nan"))
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        norm = np.linalg.norm(edge)
        if norm < 1e-12:
            continue
        u = edge / norm
        v = np.array([-u[1], u[0]])
        a = hull @ u
        b = hull @ v
        width, height = float(a.max() - a.min()), float(b.max() - b.min())
        if width * height < best[0]:
            best = (width * height, min(width, height), max(width, height))
    return best[1], best[2]


def _convex_hull(planar: np.ndarray) -> np.ndarray:
    """Convex hull by the monotone chain, counter-clockwise."""
    points = np.unique(planar, axis=0)
    if len(points) < 3:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    def turn(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(sequence):
        chain: list[np.ndarray] = []
        for p in sequence:
            while len(chain) >= 2 and turn(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        return chain[:-1]

    return np.array(half(points) + half(points[::-1]))


def rings(parts: list[Solid], link_mm: float = 180.0, minimum: int = 20) -> list[np.ndarray]:
    """Group prism solids into coil rings by connected components over their centres."""
    from scipy.spatial import cKDTree

    boxes = [s.vertices for s in parts if len(s.vertices) >= 8]
    centres = np.array([b.mean(axis=0) for b in boxes])
    if not len(centres):
        return []
    parent = list(range(len(centres)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in cKDTree(centres).query_pairs(r=link_mm):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    labels = np.array([find(i) for i in range(len(centres))])
    out = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) >= minimum:
            out.append(np.vstack([boxes[i] for i in members]))
    return out


def rib_sections(corners: np.ndarray) -> list[tuple[float, float]]:
    """(width, height) of each end rib from its corner pairwise distances, free of sweep shear."""
    if len(corners) != 8:
        return []
    centre = corners.mean(axis=0)
    shifted = corners - centre
    axis = np.linalg.svd(shifted, full_matrices=False)[2][0]
    along = shifted @ axis
    order = np.argsort(along)
    out = []
    for half in (order[:4], order[4:]):
        rib = corners[half]
        distances = np.sort(
            [
                float(np.linalg.norm(rib[i] - rib[j]))
                for i in range(4)
                for j in range(i + 1, 4)
            ]
        )
        width = float(np.mean(distances[0:2]))
        height = float(np.mean(distances[2:4]))
        diagonal = float(np.mean(distances[4:6]))
        # A rib that is not a rectangle fails Pythagoras, and a split that put corners of
        # both ribs in one half fails it too, so the check is the filter.
        if abs(diagonal - float(np.hypot(width, height))) < 0.02 * diagonal:
            out.append((width, height))
    return out


def face_planes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The brick's two end faces split along its longest axis, both required to pass the rectangle test."""
    if len(corners) != 8:
        return None
    centre = corners.mean(axis=0)
    shifted = corners - centre
    axis = np.linalg.svd(shifted, full_matrices=False)[2][0]
    order = np.argsort(shifted @ axis)
    bottom, top = corners[order[:4]], corners[order[4:]]
    for face in (bottom, top):
        distances = np.sort(
            [
                float(np.linalg.norm(face[i] - face[j]))
                for i in range(4)
                for j in range(i + 1, 4)
            ]
        )
        width = float(np.mean(distances[0:2]))
        height = float(np.mean(distances[2:4]))
        diagonal = float(np.mean(distances[4:6]))
        if abs(diagonal - float(np.hypot(width, height))) > 0.02 * diagonal:
            return None
    return bottom, top


def face_frame(face: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(unit normal, long side, short side) of one four-corner rectangle."""
    spread = face - face.mean(axis=0)
    axes = np.linalg.svd(spread, full_matrices=False)[2]
    sides = np.sort([float(np.ptp(spread @ axes[0])), float(np.ptp(spread @ axes[1]))])
    return axes[2], sides[1], sides[0]


class MeshSections:
    """Exact half-plane chord sections of a triangle mesh and its stellarator images."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=float)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        phi = np.arctan2(self.vertices[:, 1], self.vertices[:, 0])
        corner = phi[self.triangles]
        # A triangle straddling the pi branch cut would smear across the angle table;
        # its interval is widened to the whole circle instead.
        spread = corner.max(axis=1) - corner.min(axis=1)
        wraps = spread > np.pi
        self.phi_min = np.where(wraps, -np.pi, corner.min(axis=1))
        self.phi_max = np.where(wraps, np.pi, corner.max(axis=1))

    def section(
        self, angle: float, mirror: bool = False, slab_rad: float = 0.035
    ) -> np.ndarray:
        """(n, 2, 2) chord segments in (R, Z) at one angle; ``mirror`` cuts the stellarator image."""
        cut = -angle if mirror else angle
        cut = float(np.arctan2(np.sin(cut), np.cos(cut)))
        near = (self.phi_min - slab_rad <= cut) & (cut <= self.phi_max + slab_rad)
        if not near.any():
            return np.empty((0, 2, 2))
        triangles = self.triangles[near]
        corners = self.vertices[triangles]
        distance = corners[:, :, 0] * np.sin(cut) - corners[:, :, 1] * np.cos(cut)
        radial = corners[:, :, 0] * np.cos(cut) + corners[:, :, 1] * np.sin(cut)

        points, owners = [], []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            crossing = (distance[:, a] * distance[:, b]) < 0.0
            if not crossing.any():
                continue
            da, db = distance[crossing, a], distance[crossing, b]
            t = da / (da - db)
            r = radial[crossing, a] + t * (radial[crossing, b] - radial[crossing, a])
            z = (
                corners[crossing, a, 2]
                + t * (corners[crossing, b, 2] - corners[crossing, a, 2])
            )
            points.append(np.stack([r, z], axis=1))
            owners.append(np.flatnonzero(crossing))
        if not points:
            return np.empty((0, 2, 2))
        stacked = np.concatenate(points)
        owner = np.concatenate(owners)
        order = np.argsort(owner, kind="stable")
        stacked, owner = stacked[order], owner[order]
        first = np.flatnonzero(np.diff(owner, prepend=-1))
        counts = np.diff(np.append(first, len(owner)))
        starts = first[counts == 2]
        if not len(starts):
            return np.empty((0, 2, 2))
        pairs = np.stack([stacked[starts], stacked[starts + 1]], axis=1)
        pairs = pairs[(pairs[:, :, 0] > 0.0).all(axis=1)]
        if mirror:
            pairs = pairs.copy()
            pairs[:, :, 1] *= -1.0
        return pairs


def segment_displacement(
    points: np.ndarray, segments: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Displacement of each point to its nearest segment, and the distance."""
    if not len(segments):
        return np.full_like(points, np.nan), np.full(len(points), np.inf)
    start = segments[:, 0]
    span = segments[:, 1] - segments[:, 0]
    squared = np.einsum("ij,ij->i", span, span)
    squared = np.where(squared > 0.0, squared, 1.0)
    offset = points[:, None, :] - start[None, :, :]
    t = np.clip(np.einsum("pij,ij->pi", offset, span) / squared[None, :], 0.0, 1.0)
    foot = start[None, :, :] + t[:, :, None] * span[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - foot, axis=2)
    best = np.argmin(distance, axis=1)
    rows = np.arange(len(points))
    return foot[rows, best] - points, distance[rows, best]


def tube_sections(
    cloud: np.ndarray, corners: np.ndarray, bins: int = 72, minimum: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-section width and height along a closed tube, per station perpendicular to the tangent."""
    centre = cloud.mean(axis=0)
    _, _, axes = np.linalg.svd(cloud - centre, full_matrices=False)
    u, v = axes[0], axes[1]

    def angles(points):
        shifted = points - centre
        return np.arctan2(shifted @ v, shifted @ u)

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    which_cloud = np.clip(np.digitize(angles(cloud), edges) - 1, 0, bins - 1)
    which_corner = np.clip(np.digitize(angles(corners), edges) - 1, 0, bins - 1)
    centres = np.full((bins, 3), np.nan)
    for b in range(bins):
        sel = cloud[which_cloud == b]
        if len(sel) >= minimum:
            centres[b] = sel.mean(axis=0)

    widths, heights = [], []
    for b in range(bins):
        if not np.isfinite(centres[b]).all():
            continue
        after = centres[(b + 1) % bins]
        before = centres[(b - 1) % bins]
        if not (np.isfinite(after).all() and np.isfinite(before).all()):
            continue
        tangent = after - before
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            continue
        tangent /= norm
        sel = corners[which_corner == b] - centres[b]
        if len(sel) < 4:
            continue
        sel = sel - np.outer(sel @ tangent, tangent)
        basis = np.linalg.svd(sel, full_matrices=False)[2][:2]
        width, height = minimal_rectangle(sel @ basis.T)
        if np.isfinite(width):
            widths.append(width)
            heights.append(height)
    return np.array(widths), np.array(heights)
