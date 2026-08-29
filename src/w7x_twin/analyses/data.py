"""Machine-description fetch, CAD comparison, page exports, and record validation."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from w7x_twin.analyses import _common
from w7x_twin.hardware import cad, coils as coil_geometry, machine, walls
from w7x_twin.hardware.walls import base_name, load_vessel
from w7x_twin.magnetics import field, fieldlines, plasma_response
from w7x_twin.magnetics.field import VacuumField
from w7x_twin.mhd import diagnostics
from w7x_twin.mhd.equilibrium import SCAN, Twin
from w7x_twin.plasma import kinetics


# -- fetch -------------------------------------------------------------------------

# Fetch the machine-description files, pinned to upstream commits and verified by digest.
#
#     python -m w7x_twin fetch [directory]

RAW = "https://raw.githubusercontent.com"

#: Upstream commits the files below are taken from.
COMMITS = {
    "vmecpp_large_cpp_tests": "394a09a6b264057ad08e28c22fa84517e4a96d8d",
    "util-library": "9d82a89b3e947bab0d1578a64babd11bcfec8abc",
    "vmecpp": "f5dbf764fb3627cfaea72747a06a77fcc32e7938",
    "simsopt": "4dc3a3a300fc1212ff373cb7ebb0d403113bd63b",
}

_TESTS = f"{RAW}/proximafusion/vmecpp_large_cpp_tests/{COMMITS['vmecpp_large_cpp_tests']}"
_ORNL = f"{RAW}/ORNL-Fusion/util-library/{COMMITS['util-library']}"
_VMECPP = f"{RAW}/proximafusion/vmecpp/{COMMITS['vmecpp']}"
_SIMSOPT = f"{RAW}/hiddenSymmetries/simsopt/{COMMITS['simsopt']}"
_GEO = f"{_ORNL}/matlab/bfield_library_jdl/W7X"

#: Inputs this package produces rather than fetches, and what produces each.
#: A clean checkout has none of them, and every entry below is named by a command
#: that fails without it, so the roster is what a reader checks against when one
#: is missing.
GENERATED: dict[str, str] = {
    "coils.w7x_full": (
        "coils.write_extended_coils_file: coils.w7x plus the reconstructed trim "
        "and control circuits, needed by errorfield, symmetrise, strikes, koeberl"
    ),
    "coils.w7x_packcad": (
        "coils.write_finite_build_coils_file: one filament per conductor turn, "
        "needed by winding"
    ),
    "benchmarks/koeberl": (
        "the published reconstruction the koeberl benchmark is solved against, "
        "downloaded from Zenodo 8095035"
    ),
}

#: name -> (url, sha256, description)
SOURCES: dict[str, tuple[str, str, str]] = {
    "coils.w7x": (
        f"{_TESTS}/test_data/coils.w7x",
        "a1810c5f12e7ac37114b1593f5df0ef559ca7f6b82a592645d90851f28efcc3d",
        "Single-filament winding-pack model of the W7-X superconducting magnet "
        "system, with the field grid as an embedded MGRID_NLI namelist.",
    ),
    "axis_coefficients_w7x.csv": (
        f"{_TESTS}/test_data/axis_coefficients_w7x.csv",
        "e18d2a7bfbcdbb29c1a82911edc741f0c248149ac00509578c28b7286796c315",
        "Fourier coefficients of the W7-X magnetic axis.",
    ),
    "vessel.part": (
        f"{_GEO}/../vessel.part",
        "864272f7575e32e97bd24a875caa24dc84f3f312e2a4f3ce12266dcd516de415",
        "Plasma vessel contour, one field period as 41 toroidal cuts.",
    ),
    "w7x_free_bdy_vac.json": (
        f"{_VMECPP}/examples/data/w7x_free_bdy_vac.json",
        "38dd0e23ff005c1b1b272493060622ce24e9d2be92add4adc620bf39ce9a2a07",
        "Free-boundary VMEC input supplying the boundary and axis every solve "
        "starts from.",
    ),
    "w7x.json": (
        f"{_VMECPP}/examples/data/w7x.json",
        "fddb3daf1b571a5da4e229fc5712b9cfafd0cd7f51edc745ae0dfbd63afa984e",
        "Fixed-boundary VMEC input for the same configuration.",
    ),
    "reference/simsopt_W7-X.dat": (
        f"{_SIMSOPT}/src/simsopt/configs/W7-X.dat",
        "7c17c0c589026915414676fe47c10cd0aefa581c23dccbb390b08f8b73580de6",
        "Independent copy of the same filaments: an order-48 Fourier representation of "
        "coils.w7x_v001, the CAD coil set, stated to reproduce the Cartesian data to "
        "about 1e-13 m. results/archive/coil_provenance.json measures the two against each other.",
    ),
}

#: Plasma-facing components: divertor targets, baffles and the scraper element, from
#: the same source as the vessel contour.
COMPONENTS: dict[str, str] = {
    "bafhor1.t": "3c65dd87bd2fcce205c9d09099069018a30aadf30e6c9cc368cd15c1818ef4ab",
    "bafhor2.b": "d45af4988aec85c4ebe94cdde3851c1400047e6d886414c0aee4ccad94f839a5",
    "bafhormid.t": "e9f274c5e6779546204edb16bff3098d49f26a528518c62ec5cef3c63fee7bea",
    "bafver1.t": "ee5c0889c7baf4e23d6dd18a5e47061977a388b901dd42345ec99b67d0b79123",
    "bafver2.b": "81cb27bc0a5688470b53325a7b6e1227f77ed957e3d7e4f3851f4af8084e5be6",
    "bafvern8.t": "7e3e0541cc5413a4f3e88fcf677fbc6d21328f2fc1a119e29d2745a59709860d",
    "divhoran7.b": "c010044a74fd9ee4a69c41021155a11e37e8f84e92c99cda7ff0921784177243",
    "divhorn9.t": "ba84535e415f6d5c2ee73b4e2a86de4f2809872c0d1774e06d761528ae59ddb2",
    "divvern8.t": "64ec8761a47677a410abdf00adad413317e309f38e93bf0c1c60dda3f23052c9",
    "scraper_06_25_2013.t": "7282294cf67257215aec89ffbb4f2a8eb6922184b97cacdebe1774b4140b2914",
}


class DigestMismatch(RuntimeError):
    """Raised when a fetched file does not match its recorded digest."""


def fetch(url: str, destination: Path, expected: str) -> None:
    """Fetch one file and write it only if its digest matches."""
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise DigestMismatch(
            f"{destination.name}: expected sha256 {expected}, got {digest} from {url}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    print(f"  {destination.name:32s} {len(payload) / 1024:9.1f} kB  sha256 ok")


def run_fetch() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    target.mkdir(parents=True, exist_ok=True)
    print(f"writing to {target.resolve()}")
    for name, commit in COMMITS.items():
        print(f"  {name} pinned at {commit[:12]}")

    for name, (url, digest, _description) in SOURCES.items():
        fetch(url, target / name, digest)

    pfc = target / "pfc"
    pfc.mkdir(exist_ok=True)
    print(f"plasma-facing components to {pfc.resolve()}")
    for name, digest in COMPONENTS.items():
        fetch(f"{_GEO}/geo/{name}", pfc / name, digest)
    return 0


# -- prepare -----------------------------------------------------------------------

# Fetch what the package downloads, then build what it makes for itself, so a clean
# checkout reaches a state the commands can run from in one step.
#
#     python -m w7x_twin prepare [--finite-build]


def stale_generated(target: Path, path: Path) -> bool:
    """True where a generated coils file differs from what the code now produces."""
    import hashlib
    import tempfile

    from w7x_twin.hardware import coils as coil_geometry
    from w7x_twin.hardware import walls

    vessel = walls.load_vessel(target / "vessel.part")
    with tempfile.TemporaryDirectory() as work:
        probe = Path(work) / path.name
        coil_geometry.write_extended_coils_file(
            str(target / "coils.w7x"), str(probe), vessel
        )
        return (
            hashlib.sha256(probe.read_bytes()).digest()
            != hashlib.sha256(path.read_bytes()).digest()
        )


def build_generated(target: Path, finite_build: bool = False) -> list[str]:
    """Write the coils files the package generates; returns what it built."""
    from w7x_twin.hardware import coils as coil_geometry
    from w7x_twin.hardware import walls

    built: list[str] = []
    base = target / "coils.w7x"

    extended = target / "coils.w7x_full"
    # A generated input is checked against what the code generates now, not merely
    # for being present: the constants and the construction both live in coils.py,
    # and a file left from an earlier one is consumed without complaint. One built
    # before the trim mounting radius moved carried a 24 per cent error into the
    # error-field amplitude, which nothing downstream could see.
    if extended.exists() and not stale_generated(target, extended):
        print(f"  {extended.name:22s} already there and current")
    else:
        if extended.exists():
            print(f"  {extended.name:22s} stale against the current code, rebuilding")
            extended.unlink()
        vessel = walls.load_vessel(target / "vessel.part")
        circuits = coil_geometry.write_extended_coils_file(
            str(base), str(extended), vessel
        )
        print(
            f"  {extended.name:22s} {extended.stat().st_size / 1e6:6.1f} MB, "
            f"{len(circuits)} circuits"
        )
        built.append(extended.name)

    # One filament per conductor turn is 6120 filaments against 70 and tens of
    # megabytes, and `winding` builds it on demand, so it is made here only when
    # asked for.
    pack = target / "coils.w7x_packcad"
    if not finite_build:
        print(f"  {pack.name:22s} skipped; `winding` builds it, or pass --finite-build")
    elif pack.exists():
        print(f"  {pack.name:22s} already there")
    else:
        counts = coil_geometry.write_finite_build_coils_file(base, pack)
        print(
            f"  {pack.name:22s} {pack.stat().st_size / 1e6:6.1f} MB, "
            f"{sum(counts.values())} filaments"
        )
        built.append(pack.name)
    return built


def run_prepare() -> int:
    target = Path("data")
    finite_build = "--finite-build" in sys.argv

    missing = [name for name in SOURCES if not (target / name).is_file()]
    if missing or not (target / "pfc").is_dir():
        print(f"fetching {len(missing)} machine-description files")
        run_fetch()
    else:
        print(f"{len(SOURCES)} machine-description files already in {target}")

    print("\nbuilding what the package makes for itself")
    build_generated(target, finite_build)

    print()
    absent = [
        f"{name}: {why}"
        for name, why in GENERATED.items()
        if not (target / name).exists()
    ]
    for entry in absent:
        print(f"still absent, {entry}")
    if not absent:
        print("every generated input is present")
    print(
        "\nthe geometry these inputs produce:\n  "
        + str(_common.current_geometry())
    )
    return 0


# -- cad ---------------------------------------------------------------------------

# The released CAD against the geometry this package reconstructed.
# Inferred inputs measured from the released IPP STEP models (CATPRT-0886xx, AP242, mm).
#
#     python -m w7x_twin cad

CAD_RECORD = Path("results/hardware/cad_geometry.json")
ROOT = Path("data/cad")
WP = ROOT / "cs-transfer/WP_model.stp"
OUTER_VESSEL = ROOT / "12-w7-x_outer_vessel/12-W7-X outer vessel--CATPRT-088621-r001.stp"
PLASMA = ROOT / "01-W7-X-plasma/01-W7-X plasma--CATPRT-088633-r001.stp"
TRIM_RECORD = Path("results/discharges/trim_radius.json")
NON_PLANAR_SOLIDS = ROOT / "06-w7-x_non-planar_coils/06-W7-X non-planar coils--CATPRT-088628-r001.stp"
PLANAR_SOLIDS = ROOT / "07-w7-x_planar_coils/07-W7-X planar coils--CATPRT-088627-r001.stp"
VESSEL_SOLIDS = ROOT / "05-W7-X-plasma-vessel/05-W7-X plasma vessel--CATPRT-088629-r001.stp"
COILS_FILE = Path("data/coils.w7x")
VESSEL_PART = Path("data/vessel.part")
MESHES = Path("results/magnetics/w7x_machine_meshes.json")

#: Height band the trim coils sit in, in metres about the midplane.
MOUNT_BAND_M = 0.8


def digests() -> dict[str, dict]:
    """SHA256 and size of every CAD file present, which is the provenance record."""
    out = {}
    for path in sorted(ROOT.rglob("*.stp")):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        out[path.relative_to(ROOT).as_posix()] = {
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
        }
    return out


def winding_packs() -> dict:
    """Pack dimensions from the released model: turn width from brick faces,
    layer build as the distance between one brick's parallel face planes."""
    parts = cad.solids(WP)
    boxes = [s.vertices for s in parts if len(s.vertices) == 8]
    ribs = []
    for corners in boxes:
        ribs.extend(cad.rib_sections(corners))
    ribs = np.array(ribs)
    short = np.minimum(ribs[:, 0], ribs[:, 1])
    tall = np.maximum(ribs[:, 0], ribs[:, 1])
    # The model also carries stands and joints; a pack rib is one whose sides sit near
    # either population, and everything else stays out of the statistics.
    non_planar = np.abs(tall - 155.7) < 8.0
    planar = (np.abs(tall - 106.0) < 10.0) & (np.abs(short - 105.0) < 10.0)

    bounds = []
    for corners in cad.rings(parts):
        widths, heights = cad.tube_sections(corners, corners, bins=64)
        if len(widths) >= 20:
            bounds.append(float(np.median(np.maximum(widths, heights))))

    builds: dict[str, list[float]] = {"non_planar": [], "planar": []}
    for corners in boxes:
        split = cad.face_planes(np.asarray(corners, dtype=float))
        if split is None:
            continue
        bottom, top = split
        normal_b, long_b, short_b = cad.face_frame(bottom)
        normal_t, long_t, short_t = cad.face_frame(top)
        alignment = float(np.dot(normal_b, normal_t))
        if abs(alignment) < 0.99:
            continue
        normal = normal_b + np.sign(alignment) * normal_t
        normal /= np.linalg.norm(normal)
        build = abs(float((top.mean(axis=0) - bottom.mean(axis=0)) @ normal))
        if all(150.0 < side < 162.0 for side in (long_b, long_t)) and all(
            70.0 < side < 120.0 for side in (short_b, short_t)
        ):
            builds["non_planar"].append(build)
        elif all(100.0 < side < 112.0 for side in (long_b, long_t)) and all(
            95.0 < side < 112.0 for side in (short_b, short_t)
        ):
            builds["planar"].append(build)

    record = {
        "solids": len(parts),
        "ribs": int(len(ribs)),
        "non_planar_ribs": int(non_planar.sum()),
        "non_planar_turn_direction_mm": float(np.median(tall[non_planar])),
        "non_planar_turn_direction_spread_mm": float(np.ptp(tall[non_planar])),
        "non_planar_layer_direction_bound_mm": float(np.max(bounds)) if bounds else None,
        "planar_ribs": int(planar.sum()),
        "planar_rib_mm": [
            float(np.median(short[planar])),
            float(np.median(tall[planar])),
        ],
        "package_non_planar_mm": [
            1e3 * ((coil_geometry.NON_PLANAR_PACK.turns_per_layer - 1)
                   * coil_geometry.NON_PLANAR_PACK.pitch_across_turns
                   + coil_geometry.CONDUCTOR_SIZE_M),
            1e3 * ((coil_geometry.NON_PLANAR_PACK.layers - 1)
                   * coil_geometry.NON_PLANAR_PACK.pitch
                   + coil_geometry.CONDUCTOR_SIZE_M),
        ],
    }
    for kind, values in builds.items():
        if len(values) >= 100:
            arr = np.array(values)
            record[f"{kind}_layer_direction_mm"] = float(np.median(arr))
            record[f"{kind}_layer_direction_spread_mm"] = float(
                np.percentile(arr, 95) - np.percentile(arr, 5)
            )
            record[f"{kind}_bricks"] = int(len(arr))
    return record


def coil_centrelines() -> dict:
    """Filament offsets from the released coil solids, measured per sample against the
    local point-cloud box midpoint perpendicular to the filament tangent."""
    from scipy.spatial import cKDTree

    from w7x_twin.hardware import machine

    coils = machine.load_coils(COILS_FILE)
    filaments = [np.asarray(f, dtype=float) for group in coils.filaments for f in group]

    # The B-rep control nets are one point per forty square centimetres, far too sparse
    # for a local midpoint, so the clouds are the meshed release: the same files through
    # the tessellator, whose vertices lie on the surfaces at a uniform density.
    import base64

    manifest = json.loads(MESHES.read_text())
    clouds = []
    for name in ("non_planar_coils", "planar_coils"):
        block = manifest["components"].get(name)
        if block is None:
            continue
        clouds.append(
            np.frombuffer(
                base64.b64decode(block["positions_b64"]), dtype=np.float32
            ).reshape(-1, 3).astype(np.float64)
        )

    trees = [cKDTree(cloud) for cloud in clouds]
    rows = []
    for index, filament in enumerate(filaments):
        samples = filament[::4]
        best = None
        for cloud, tree in zip(clouds, trees, strict=True):
            # The ball must hold the whole section, or its midpoint leans toward
            # the near walls: the pack diagonal is 137 mm from centre plus mesh
            # coarseness, so 210 mm covers it at every station.
            near = tree.query_ball_point(samples, r=0.21)
            covered = np.array([len(n) >= 8 for n in near])
            if covered.mean() < 0.6:
                continue
            offsets = []
            for k, members in enumerate(near):
                if len(members) < 8:
                    continue
                local = cloud[members]
                midpoint = 0.5 * (local.min(axis=0) + local.max(axis=0))
                after = samples[(k + 1) % len(samples)]
                before = samples[k - 1]
                tangent = after - before
                tangent /= max(np.linalg.norm(tangent), 1e-12)
                offset = midpoint - samples[k]
                offset -= (offset @ tangent) * tangent
                offsets.append(float(np.linalg.norm(offset)))
            if not offsets:
                continue
            row = {
                "filament": index,
                "stations": len(offsets),
                "coverage": float(covered.mean()),
                "median_mm": 1e3 * float(np.median(offsets)),
                "p95_mm": 1e3 * float(np.percentile(offsets, 95)),
            }
            if best is None or row["median_mm"] < best["median_mm"]:
                best = row
        if best is not None:
            rows.append(best)

    medians = [r["median_mm"] for r in rows]
    return {
        "solids_measured": len(clouds),
        "filaments_matched": len(rows),
        "median_mm": float(np.median(medians)) if medians else float("nan"),
        "worst_median_mm": float(np.max(medians)) if medians else float("nan"),
        "per_filament": rows,
    }


def vessel_against_part() -> dict:
    """vessel.part contours against the released plasma-vessel cloud, one-sided, cut by cut."""
    from scipy.spatial import cKDTree

    from w7x_twin.hardware import walls

    part = walls.load_vessel(VESSEL_PART)
    points = cad.all_points(VESSEL_SOLIDS) / 1e3
    radius = np.hypot(points[:, 0], points[:, 1])
    height = points[:, 2]
    folded = np.mod(np.arctan2(points[:, 1], points[:, 0]), part.period)

    per_cut = []
    for index in range(len(part.phi)):
        selected = np.abs(folded - float(part.phi[index])) < np.radians(0.35)
        if int(selected.sum()) < 50:
            continue
        tree = cKDTree(np.stack([radius[selected], height[selected]], axis=1))
        contour = np.stack([part.r[index], part.z[index]], axis=1)
        distance, _ = tree.query(contour)
        per_cut.append(
            {
                "phi_degrees": float(np.degrees(part.phi[index])),
                "median_mm": 1e3 * float(np.median(distance)),
                "p95_mm": 1e3 * float(np.percentile(distance, 95)),
            }
        )
    return {
        "cuts": len(per_cut),
        "median_mm": float(np.median([c["median_mm"] for c in per_cut])),
        "p95_mm": float(np.median([c["p95_mm"] for c in per_cut])),
        "worst_cut_mm": float(np.max([c["p95_mm"] for c in per_cut])),
        "per_cut": per_cut,
    }


def components_against_release() -> dict:
    """Component contours against the released divertor and baffle surfaces, per cut plane
    against tessellation chords, uncovered where the release carries no counterpart."""
    import base64

    from w7x_twin.hardware import walls

    manifest = json.loads(MESHES.read_text())
    sections = {}
    for name in ("divertor", "baffle", "heat_shield"):
        block = manifest["components"].get(name)
        if block is None:
            continue
        vertices = np.frombuffer(
            base64.b64decode(block["positions_b64"]), dtype=np.float32
        ).reshape(-1, 3).astype(np.float64)
        triangles = np.frombuffer(
            base64.b64decode(block["indices_b64"]), dtype=np.uint32
        ).reshape(-1, 3).astype(np.int64)
        sections[name] = cad.MeshSections(vertices, triangles)
    if not sections:
        return {"per_component": []}

    period = 2.0 * np.pi / 5.0
    tolerance = 0.06
    rows = []
    for component in walls.load_components("data/pfc"):
        distances: list[float] = []
        covered = 0
        total = 0
        for cut in range(len(component.phi)):
            contour = np.stack([component.r[cut], component.z[cut]], axis=1)
            best = None
            for source in sections.values():
                for k in range(5):
                    for mirror in (False, True):
                        angle = (
                            (-1.0 if mirror else 1.0) * float(component.phi[cut])
                            + k * period
                        )
                        segments = source.section(angle, mirror)
                        _, distance = cad.segment_displacement(contour, segments)
                        valid = distance <= tolerance
                        score = (int(valid.sum()), -float(np.sum(distance[valid])))
                        if best is None or score > best[0]:
                            best = (score, distance, valid)
            _, distance, valid = best
            total += len(contour)
            covered += int(valid.sum())
            distances.extend(distance[valid].tolist())
        rows.append(
            {
                "component": component.name,
                "coverage": covered / max(total, 1),
                "median_mm": 1e3 * float(np.median(distances)) if distances else float("nan"),
                "p95_mm": 1e3 * float(np.percentile(distances, 95)) if distances else float("nan"),
            }
        )
    return {"per_component": rows}


def outer_vessel_radius() -> dict:
    """Outboard outer-vessel radius over the trim-coil mounting band, shell alone."""
    points = cad.all_points(OUTER_VESSEL) / 1e3
    radius = np.hypot(points[:, 0], points[:, 1])
    band = np.abs(points[:, 2]) < MOUNT_BAND_M
    phi = np.arctan2(points[band, 1], points[band, 0])
    r_band = radius[band]
    edges = np.linspace(-np.pi, np.pi, 73)
    outboard = []
    for low, high in zip(edges[:-1], edges[1:]):
        sel = (phi >= low) & (phi < high)
        if sel.sum() >= 5:
            outboard.append(float(np.max(r_band[sel])))
    return {
        "points": int(len(points)),
        "outboard_radius_m": {
            "median": float(np.median(outboard)),
            "p5": float(np.percentile(outboard, 5)),
            "p95": float(np.percentile(outboard, 95)),
        },
        "band_m": MOUNT_BAND_M,
    }


def plasma_boundary() -> dict | None:
    """Released plasma-surface points against the VMEC boundary at each point's own toroidal angle."""
    try:
        from w7x_twin.mhd import diagnostics
    except ImportError:
        return None

    points = cad.all_points(PLASMA) / 1e3
    radius = np.hypot(points[:, 0], points[:, 1])
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    height = points[:, 2]

    twin = _common.twin()
    equilibrium = twin.solve(twin.state("standard"), SCAN)
    surface = int(equilibrium.wout.ns) - 1

    residuals = np.empty(len(points))
    for index in range(len(points)):
        r_curve, z_curve = diagnostics.flux_surface(
            equilibrium.wout, surface, float(phi[index]), 256
        )
        residuals[index] = float(
            np.min(np.hypot(r_curve - radius[index], z_curve - height[index]))
        )

    # Where the tail lives: the residual binned over the field period's own angle and the
    # bean-to-triangle span, so print smoothing at the tips and a configuration difference
    # in the bulk read differently.
    folded = np.mod(phi, 2.0 * np.pi / 5.0)
    edges = np.linspace(0.0, 2.0 * np.pi / 5.0, 10)
    by_angle = []
    for low, high in zip(edges[:-1], edges[1:]):
        sel = (folded >= low) & (folded < high)
        if sel.sum() >= 20:
            by_angle.append(
                {
                    "phi_degrees": float(np.degrees(0.5 * (low + high))),
                    "median_mm": 1e3 * float(np.median(residuals[sel])),
                    "p95_mm": 1e3 * float(np.percentile(residuals[sel], 95)),
                }
            )
    tail = residuals >= np.percentile(residuals, 90)
    return {
        "points": int(len(points)),
        "geometry": twin.geometry.as_dict(),
        "residual_mm": {
            "median": 1e3 * float(np.median(residuals)),
            "p95": 1e3 * float(np.percentile(residuals, 95)),
            "worst": 1e3 * float(np.max(residuals)),
        },
        "by_angle": by_angle,
        "tail_phi_degrees": [
            float(v) for v in np.degrees(folded[tail])
        ],
        "tail_z_m": [float(v) for v in height[tail]],
    }


def run_cad() -> int:
    record: dict = {"files": digests()}
    stored = json.loads(CAD_RECORD.read_text()) if CAD_RECORD.exists() else {}
    print(f"{len(record['files'])} CAD files under {ROOT}")

    if WP.exists():
        packs = winding_packs()
        record["winding_packs"] = packs
        modelled = packs["package_non_planar_mm"]
        print(
            f"winding-pack model: {packs['solids']} prisms, {packs['ribs']} exact ribs; "
            f"the non-planar turn direction is "
            f"{packs['non_planar_turn_direction_mm']:.1f} mm to a spread of "
            f"{packs['non_planar_turn_direction_spread_mm']:.1f}, the layer direction is "
            f"bounded at or below {packs['non_planar_layer_direction_bound_mm']:.1f}, and "
            f"the package's pack is {modelled[0]:.1f} x {modelled[1]:.1f}"
        )
        if "non_planar_layer_direction_mm" in packs:
            print(
                f"  between the brick face planes the non-planar layer build is "
                f"{packs['non_planar_layer_direction_mm']:.2f} mm over "
                f"{packs['non_planar_bricks']} bricks to a 5-95 spread of "
                f"{packs['non_planar_layer_direction_spread_mm']:.2f}, and the planar "
                f"is {packs['planar_layer_direction_mm']:.2f}"
            )
        print(
            f"  the planar ribs measure {packs['planar_rib_mm'][0]:.1f} x "
            f"{packs['planar_rib_mm'][1]:.1f} mm"
        )

    if OUTER_VESSEL.exists():
        vessel = outer_vessel_radius()
        record["outer_vessel"] = vessel
        stated = vessel["outboard_radius_m"]
        print(
            f"outer vessel: outboard radius {stated['median']:.3f} m "
            f"({stated['p5']:.3f} to {stated['p95']:.3f}) within "
            f"{MOUNT_BAND_M:.1f} m of the midplane"
        )
        if TRIM_RECORD.exists():
            trim = json.loads(TRIM_RECORD.read_text())
            band = trim.get("pinned_radius_band_m", [float("nan")] * 2)
            print(
                f"  the error-field measurement pinned the trim mounting radius to "
                f"{trim.get('pinned_radius_m', float('nan')):.3f} m "
                f"({band[0]:.3f} to {band[1]:.3f}), against the reconstruction's "
                f"{trim.get('reconstruction_radius_m', float('nan')):.1f}"
            )
            record["trim_radius_pinned_m"] = trim.get("pinned_radius_m")

    if MESHES.exists() and COILS_FILE.exists():
        centres = coil_centrelines()
        record["coil_centrelines"] = centres
        print(
            f"coil solids: {centres['solids_measured']} windings measured, "
            f"{centres['filaments_matched']} filaments matched; the filaments sit "
            f"{centres['median_mm']:.1f} mm from the released centrelines in the median, "
            f"{centres['worst_median_mm']:.1f} at the worst coil"
        )

    if VESSEL_SOLIDS.exists() and VESSEL_PART.exists():
        vessel_check = vessel_against_part()
        record["vessel_part"] = vessel_check
        print(
            f"vessel.part: {vessel_check['cuts']} cuts sit "
            f"{vessel_check['median_mm']:.1f} mm from the released vessel in the median, "
            f"{vessel_check['p95_mm']:.1f} at the 95th percentile, "
            f"{vessel_check['worst_cut_mm']:.1f} at the worst cut"
        )

    if MESHES.exists() and Path("data/pfc").exists():
        released = components_against_release()
        record["components_release"] = released
        for row in released["per_component"]:
            print(
                f"  {row['component']:38s} {100 * row['coverage']:5.1f} % covered, "
                f"{row['median_mm']:6.1f} mm median, {row['p95_mm']:6.1f} at p95"
            )

    boundary = plasma_boundary() if PLASMA.exists() else None
    if boundary is not None:
        record["plasma_boundary"] = boundary
        residual = boundary["residual_mm"]
        print(
            f"plasma model: {boundary['points']} CAD points sit "
            f"{residual['median']:.1f} mm from the twin's boundary in the median, "
            f"{residual['p95']:.1f} at the 95th percentile"
        )
    elif PLASMA.exists():
        print("plasma model present; the boundary comparison needs the solver")

    # A section whose inputs are absent on this host keeps its stored value, so the
    # record refreshes from either side of the repo's CAD-and-solver split. The stamp
    # is not carried over: it belongs to this run, and a stored one taken forward
    # would outlive the geometry it names.
    for key, value in stored.items():
        if key == "geometry":
            continue
        record.setdefault(key, value)

    # The comparison is of the released CAD against this package's own geometry, so
    # the record names which version of it was compared.
    _common.write_record(CAD_RECORD, record, geometry=_common.current_geometry())
    return 0


# -- cut-contours ------------------------------------------------------------------

# Recut the component contours onto exact half-plane sections of the released meshes.
#
#     python tools/cut_target_contours.py [--write] [component ...]

PFC = Path("data/pfc")

#: Component -> released mesh block the surface lives in.
CONTOUR_SOURCES = {
    "divhorn9.t": "divertor",
    "divvern8.t": "divertor",
    "divhoran7.b": "divertor",
    "bafhor1.t": "baffle",
    "bafhor2.b": "baffle",
    "bafhormid.t": "baffle",
    "bafver1.t": "baffle",
    "bafver2.b": "baffle",
    "bafvern8.t": "baffle",
}

#: A vertex further than this from every intersection segment has no released
#: counterpart at that cut and takes its displacement from its neighbours.
TOLERANCE_M = 0.06
NUM_FIELD_PERIODS = 5


def load_mesh(name: str) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(MESHES.read_text())
    block = manifest["components"][name]
    vertices = np.frombuffer(
        base64.b64decode(block["positions_b64"]), dtype=np.float32
    ).reshape(-1, 3).astype(np.float64)
    triangles = np.frombuffer(
        base64.b64decode(block["indices_b64"]), dtype=np.uint32
    ).reshape(-1, 3).astype(np.int64)
    return vertices, triangles


def smooth_along(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Displacement field filtered along the contour: nearest-fill, median, then boxcar."""
    n = len(values)
    out = values.copy()
    if valid.any() and not valid.all():
        indices = np.arange(n)
        out[~valid] = np.interp(
            indices[~valid], indices[valid], values[valid]
        )
    width = max(3, min(5, (n // 6) * 2 + 1))
    half = width // 2
    padded = np.pad(out, (half, half), mode="edge")
    medianed = np.array(
        [np.median(padded[i : i + width]) for i in range(n)]
    )
    padded = np.pad(medianed, (half, half), mode="edge")
    kernel = np.ones(width) / width
    return np.convolve(padded, kernel, mode="valid")


def recut(filename: str, cutter: cad.MeshSections, write: bool) -> dict:
    path = PFC / filename
    component = walls.load_component(path, filename)
    period = 2.0 * np.pi / NUM_FIELD_PERIODS

    # One toroidal image serves the whole component: choosing per cut lets the mirror
    # image win a cut on tessellation noise and seam the surface between cuts.
    images = [
        (k, mirror) for k in range(NUM_FIELD_PERIODS) for mirror in (False, True)
    ]
    totals = []
    cached: dict[tuple[int, int, bool], tuple[np.ndarray, np.ndarray]] = {}
    for k, mirror in images:
        count, closeness = 0, 0.0
        for cut in range(len(component.phi)):
            contour = np.stack([component.r[cut], component.z[cut]], axis=1)
            angle = (-1.0 if mirror else 1.0) * component.phi[cut] + k * period
            segments = cutter.section(float(angle), mirror)
            displacement, distance = cad.segment_displacement(contour, segments)
            cached[(cut, k, mirror)] = (displacement, distance)
            valid = distance <= TOLERANCE_M
            count += int(valid.sum())
            closeness -= float(np.sum(distance[valid]))
        totals.append((count, closeness))
    k, mirror = images[int(np.argmax([t[0] + 1e-9 * t[1] for t in totals]))]

    applied, residual, coverage = [], [], []
    new_r = component.r.copy()
    new_z = component.z.copy()
    for cut in range(len(component.phi)):
        displacement, distance = cached[(cut, k, mirror)]
        valid = distance <= TOLERANCE_M
        if not valid.any():
            coverage.append(0.0)
            continue
        coverage.append(float(valid.mean()))
        dr = smooth_along(displacement[:, 0], valid)
        dz = smooth_along(displacement[:, 1], valid)
        new_r[cut] = component.r[cut] + dr
        new_z[cut] = component.z[cut] + dz
        applied.extend(np.hypot(dr, dz).tolist())
        residual.extend(distance[valid].tolist())

    report = {
        "component": filename,
        "cuts": len(component.phi),
        "coverage": float(np.mean(coverage)) if coverage else 0.0,
        "applied_median_mm": 1e3 * float(np.median(applied)) if applied else float("nan"),
        "applied_p95_mm": 1e3 * float(np.percentile(applied, 95)) if applied else float("nan"),
        "raw_residual_median_mm": 1e3 * float(np.median(residual)) if residual else float("nan"),
    }
    if write and applied:
        rewrite(path, component, new_r, new_z)
    return report


def rewrite(path: Path, component, new_r: np.ndarray, new_z: np.ndarray) -> None:
    """The same file with the same layout, carrying the released surface."""
    lines = path.read_text().split("\n")
    out = lines[:2]
    cursor = 2
    for cut in range(len(component.phi)):
        out.append(lines[cursor])
        cursor += 1
        for point in range(component.r.shape[1]):
            columns = lines[cursor].split()
            if len(columns) < 2:
                raise ValueError(f"{path.name}: point row {cursor} has {len(columns)} columns")
            columns[0] = f"{1e2 * new_r[cut, point]:.3f}"
            columns[1] = f"{1e2 * new_z[cut, point]:.3f}"
            out.append("  " + " ".join(f"{c:>10s}" for c in columns))
            cursor += 1
    out.extend(lines[cursor:])
    path.write_text("\n".join(out))


def run_cut_contours() -> int:
    if not MESHES.is_file():
        raise SystemExit(f"no mesh manifest at {MESHES}; run tools/tessellate_cad.py")
    write = "--write" in sys.argv
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CONTOUR_SOURCES)

    meshes = {}
    for filename in wanted:
        source = CONTOUR_SOURCES[filename]
        if source not in meshes:
            vertices, triangles = load_mesh(source)
            meshes[source] = cad.MeshSections(vertices, triangles)
            print(f"{source}: {len(triangles)} triangles")

    layout = _common.Table(
        ("component", ">16s"), ("cuts", "5d"), ("covered", ">8s"),
        ("moved [mm]", "11.1f"), ("p95", "7.1f"), ("raw residual", "13.1f"),
    )
    print()
    layout.begin()
    for filename in wanted:
        report = recut(filename, meshes[CONTOUR_SOURCES[filename]], write)
        layout.row(
            report["component"], report["cuts"], f"{100 * report['coverage']:7.1f}%",
            report["applied_median_mm"], report["applied_p95_mm"],
            report["raw_residual_median_mm"],
        )
    if write:
        print("\nrewrote the component files in place")
    return 0

# -- page encoding -----------------------------------------------------------------

# Whole-torus geometry bundle for the renderer, float32 base64 arrays in Cartesian metres.

def encode(array: np.ndarray, dtype=np.float32) -> dict:
    """One array as a base64 payload with its shape, for a renderer to unpack."""
    data = np.ascontiguousarray(array, dtype=dtype)
    return {
        "dtype": np.dtype(dtype).name,
        "shape": list(data.shape),
        "data": base64.b64encode(data.tobytes()).decode(),
    }


def cylindrical_to_cartesian(r: np.ndarray, phi: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(..., 3) Cartesian points from cylindrical components that broadcast together."""
    r, phi, z = np.broadcast_arrays(
        np.asarray(r, dtype=float), np.asarray(phi, dtype=float), np.asarray(z, dtype=float)
    )
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def surface_grid(r: np.ndarray, phi: np.ndarray, z: np.ndarray) -> dict:
    """Closed tube surface from (n_phi, n_theta) grids, wrapping in both directions."""
    r = np.asarray(r, dtype=float)
    num_phi, num_theta = r.shape
    vertices = cylindrical_to_cartesian(
        r, np.asarray(phi, dtype=float)[:, None], np.asarray(z, dtype=float)
    ).reshape(-1, 3)

    i = np.arange(num_phi)[:, None]
    j = np.arange(num_theta)[None, :]
    a = i * num_theta + j
    b = ((i + 1) % num_phi) * num_theta + j
    c = ((i + 1) % num_phi) * num_theta + (j + 1) % num_theta
    d = i * num_theta + (j + 1) % num_theta
    triangles = np.concatenate(
        [np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)], axis=1
    ).reshape(-1, 3)
    return {"vertices": encode(vertices), "triangles": encode(triangles, np.uint32)}


def vessel_surface(vessel, num_phi_per_period: int = 24) -> dict:
    """The plasma vessel over the whole torus, from its one-period contour."""
    period = vessel.period
    phi = np.linspace(
        0.0, 2.0 * np.pi, num_phi_per_period * vessel.num_field_periods, endpoint=False
    )
    cuts = [vessel.cut_at(float(np.mod(angle, period))) for angle in phi]
    r = np.array([cut[0] for cut in cuts])
    z = np.array([cut[1] for cut in cuts])
    return surface_grid(r, phi, z)


def component_surfaces(components, num_field_periods: int = 5) -> list[dict]:
    """Each plasma-facing component as an open strip over the torus, tagged by module and unit."""
    out: list[dict] = []
    period = 2.0 * np.pi / num_field_periods
    for component in components:
        base_phi = np.asarray(component.phi, dtype=float)
        for module in range(num_field_periods):
            for mirrored in (False, True):
                phi = base_phi + module * period
                z = component.z
                if mirrored:
                    phi = -base_phi + module * period
                    z = -component.z
                vertices = cylindrical_to_cartesian(
                    component.r, phi[:, None], z
                ).reshape(-1, 3)
                num_cuts, num_points = component.r.shape
                i = np.arange(num_cuts - 1)[:, None]
                j = np.arange(num_points - 1)[None, :]
                a = i * num_points + j
                b = (i + 1) * num_points + j
                c = (i + 1) * num_points + j + 1
                d = i * num_points + j + 1
                triangles = np.concatenate(
                    [np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)], axis=1
                ).reshape(-1, 3)
                out.append(
                    {
                        # The unit carries the upper or lower qualifier, so the name
                        # does not repeat it and does not contradict it on the image.
                        "name": base_name(component.name),
                        "module": module + 1,
                        "unit": "lower" if mirrored else "upper",
                        "vertices": encode(vertices),
                        "triangles": encode(triangles, np.uint32),
                    }
                )
    return out


def coil_polylines(coils) -> list[dict]:
    """Every coil filament as a closed polyline, grouped by circuit."""
    out: list[dict] = []
    for key, group in zip(coils.circuit_keys, coils.filaments, strict=True):
        for index, xyz in enumerate(group):
            out.append({"circuit": key, "index": index, "points": encode(xyz)})
    return out


def field_line_paths(section, stride: int = 1) -> list[dict]:
    """Traced field lines as Cartesian polylines, dropping the parts after a strike."""
    if section.path is None:
        return []
    samples = section.path[::stride]
    out: list[dict] = []
    for line in range(samples.shape[1]):
        r, phi, z = samples[:, line, 0], samples[:, line, 1], samples[:, line, 2]
        alive = np.isfinite(r) & np.isfinite(z)
        if alive.sum() < 2:
            continue
        points = cylindrical_to_cartesian(r[alive], phi[alive], z[alive])
        out.append({"line": line, "points": encode(points)})
    return out


def write_bundle(path: str | Path, payload: dict) -> Path:
    """Write the bundle as JSON and report its size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


# -- export-geometry ---------------------------------------------------------------

# Write results/magnetics/w7x_geometry.json; flux surfaces are traced by the page, not stored.

GEOMETRY_OUT = Path("results/magnetics/w7x_geometry.json")

#: The coils file to draw from. The extended one carries the trim and control circuits the
#: page can energise, so the filaments drawn are the filaments those sliders drive.
GEOMETRY_COILS_FILE = "coils.w7x_full"

#: Field lines traced through the island region, and how far.
ISLAND_LINES = 28
ISLAND_TURNS = 20
PATH_EVERY = 4


def run_export_geometry() -> int:
    key = _common.arg(1, default="standard")
    coils_file = GEOMETRY_COILS_FILE if (Path("data") / GEOMETRY_COILS_FILE).exists() else "coils.w7x"
    twin = _common.twin(coils_file=coils_file)
    config = machine.get(key)
    print(f"{config.label}, filaments from {coils_file}")
    print(f"  {twin.geometry}")

    state = twin.state(key)
    equilibrium = twin.solve(state, SCAN)
    wout = equilibrium.wout
    vacuum = VacuumField(twin.response, state.currents)
    vessel = _common.vessel()
    elements = _common.components()

    # Field lines through the island chain, kept in three dimensions rather than
    # reduced to a section, and stopped where they reach a component.
    r_axis, z_axis = fieldlines.find_axis(vacuum)
    r_lcfs, _ = diagnostics.boundary_cut(wout, 0.0)
    half_width = r_lcfs.max() - r_axis
    starts = r_axis + np.linspace(1.0, 1.32, ISLAND_LINES) * half_width

    started = time.monotonic()
    section, _ = fieldlines.trace(
        vacuum,
        starts,
        np.full(starts.shape, z_axis),
        turns=ISLAND_TURNS,
        plane_phi=0.0,
        vessel=vessel,
        components=elements,
        record_path_every=PATH_EVERY,
    )
    lines = field_line_paths(section)
    print(
        f"  {len(lines)} field lines x {ISLAND_TURNS} turns in "
        f"{time.monotonic() - started:.1f} s"
    )

    strikes = section.strikes
    hit = np.flatnonzero(strikes.struck)
    strike_points = cylindrical_to_cartesian(
        strikes.r[hit], strikes.phi[hit], strikes.z[hit]
    )
    module, is_upper = walls.unit_of(strikes.phi[hit], strikes.z[hit])
    payload = {
        "geometry": twin.geometry.as_dict(),
        "configuration": {
            "key": key,
            "label": config.label,
            "extcur": [float(value) for value in config.currents],
            "iota_axis": float(np.asarray(wout.iotaf)[0]),
            "iota_edge": float(np.asarray(wout.iotaf)[-1]),
            # The transform profile, with the island chains it runs against: where it
            # meets one, the divertor's strike lines sit.
            "iota_profile": [float(value) for value in np.asarray(wout.iotaf)],
            "resonances": [
                {"label": f"{n}/{m}", "value": n / m}
                for n, m in diagnostics.DIVERTOR_RESONANCES
            ],
            "b_axis": float(wout.b0),
            "volume": float(wout.volume_p),
            "major_radius": float(wout.Rmajor_p),
            "minor_radius": float(wout.Aminor_p),
        },
        "coils": coil_polylines(twin.coils),
        "vessel": vessel_surface(vessel),
        "components": component_surfaces(elements),
        "field_lines": lines,
        "strikes": {
            "points": encode(strike_points),
            "connection_length": encode(strikes.connection_length_m[hit]),
            "module": encode(module, np.int32),
            "upper": encode(is_upper.astype(np.uint8), np.uint8),
        },
    }

    path = write_bundle(GEOMETRY_OUT, payload)
    print(
        f"wrote {path} ({path.stat().st_size / 1024 / 1024:.2f} MB): "
        f"{len(payload['coils'])} coils, {len(payload['components'])} component "
        f"instances, {len(lines)} field lines, {len(hit)} strikes"
    )
    return 0


# -- export-field ------------------------------------------------------------------

# Per-circuit unit-current response plus a beta-weighted plasma block, coarsened only
# as far as the strike positions stay inside the current-driven migration.

FIELD_OUT = Path("results/magnetics/w7x_field.json")
#: Interpreter carrying a CUDA-capable torch for the out-of-process Biot-Savart
#: worker, from the environment; unset leaves the summation on the CPU.
GPU_PYTHON = os.environ.get("W7X_TWIN_GPU_PYTHON", "")

#: Coarsenings of the field grid tried, in (phi, Z, R), coarsest first. The stride in phi is
#: applied per field period. The one shipped is the coarsest whose tracer resolves the strike
#: line's own migration, so the payload is set by what the page has to answer.
CANDIDATE_STRIDES = ((2, 2, 2), (2, 1, 1), (1, 2, 2), (1, 1, 1))
#: Distance the strike line moves across the bootstrap current range, which is what the
#: page has to resolve for its tracer to carry exhaust. Measured by the migration scan.
STRIKE_MIGRATION_M = 0.122

#: Volume sampling of the plasma current. The island region sits outside the plasma, where
#: the integrand is regular; ``python -m w7x_twin response`` bounds the residual and carries
#: the boundary by virtual casing, where the volume integral does not converge at all.
PLASMA_THETA, PLASMA_ZETA, PLASMA_STRIDE = 80, 240, 2

#: The coils file carrying the trim and control circuits alongside the superconducting set.
AUXILIARY_COILS = "coils.w7x_full"
#: Grid the auxiliary response is tabulated on: coarse, and over the whole torus with no
#: symmetry assumed, since one energised trim coil is periodic in neither the field period
#: nor the midplane. Their field is smooth on a metre scale, so this resolves it.
AUXILIARY_R, AUXILIARY_Z, AUXILIARY_PHI_PER_PERIOD = 31, 31, 9

#: Configurations offered as presets, in circuit order.
PRESETS = (
    "standard",
    "high_mirror_ref167",
    "op12a_22ka_mimic",
    "op2_22ka",
    "narrow_mirror",
)


#: Current per turn each auxiliary circuit is rated for.
AUXILIARY_LIMITS = {
    "trim_a1": 1800.0, "trim_a2": 1800.0, "trim_a3": 1800.0, "trim_a4": 1800.0,
    "trim_b1": 1950.0,
    **{f"cc{m}{side}": 2500.0 for m in range(1, 6) for side in "ul"},
}


def auxiliary_block(twin: Twin, cache_dir) -> dict | None:
    """Trim and control response on its own whole-torus grid, the type A block rotated
    between modules and the rotation checked against direct computation."""
    if all(key in machine.CIRCUIT_ORDER for key in twin.coils.circuit_keys):
        print("the loaded coils file carries no auxiliary circuits")
        return None

    grid = dataclasses.replace(
        field.full_torus_grid(twin.coils.grid, AUXILIARY_PHI_PER_PERIOD),
        num_r=AUXILIARY_R,
        num_z=AUXILIARY_Z,
    )
    if grid.num_phi % 5:
        raise ValueError(
            f"{grid.num_phi} toroidal points is not a multiple of five, so a 72 degree "
            "rotation is not a whole number of grid points"
        )
    table = field.build_response_table(
        twin.coils, grid=grid, cache_dir=cache_dir, verbose=True
    )
    shape = (grid.num_phi, grid.num_z, grid.num_r)
    stacked = {
        name: np.asarray(source).reshape((twin.coils.num_circuits, *shape))
        for name, source in (("b_r", table.b_r), ("b_p", table.b_p), ("b_z", table.b_z))
    }

    keys = list(twin.coils.circuit_keys)
    auxiliary = [key for key in keys if key not in machine.CIRCUIT_ORDER]
    #: One stored block per distinct winding. Every other circuit of the same winding is
    #: that block rotated by whole modules, which is what a rigid rotation about the machine
    #: axis does to cylindrical components without mixing them.
    stored = [key for key in ("trim_a1", "trim_b1", "cc1u", "cc1l") if key in keys]
    rotation = grid.num_phi // 5

    def source_of(key: str) -> tuple[str, int]:
        """The stored block a circuit is a rotation of, and by how many modules."""
        if key.startswith("trim_a"):
            return "trim_a1", int(key[-1]) - 1
        if key.startswith("cc"):
            return f"cc1{key[-1]}", int(key[2]) - 1
        return key, 0

    worst = 0.0
    checked = 0
    for key in auxiliary:
        block, modules = source_of(key)
        if modules == 0 or block not in keys:
            continue
        checked += 1
        for name in stacked:
            direct = stacked[name][keys.index(key)]
            rotated = np.roll(stacked[name][keys.index(block)], modules * rotation, 0)
            scale = float(np.max(np.abs(direct)))
            worst = max(worst, float(np.max(np.abs(direct - rotated)) / max(scale, 1e-30)))
    print(
        f"{checked} circuits reproduce their own block under a {rotation}-point rotation "
        f"to {worst:.2e} of their own amplitude"
    )

    blocks = {
        name: encode(np.stack([stacked[name][keys.index(k)] for k in stored]))
        for name in stacked
    }
    turns = dict(zip(keys, (int(v) for v in twin.coils.turns()), strict=True))
    print(
        f"auxiliary response {len(auxiliary)} circuits on {shape} from "
        f"{len(stored)} stored blocks: " + ", ".join(auxiliary)
    )
    return {
        "grid": {
            "r_min": grid.r_min,
            "r_max": grid.r_max,
            "num_r": grid.num_r,
            "z_min": grid.z_min,
            "z_max": grid.z_max,
            "num_z": grid.num_z,
            "num_phi": grid.num_phi,
            "num_field_periods": 1,
        },
        "rotation_points": rotation,
        "rotation_residual": worst,
        "blocks": stored,
        "circuits": [
            {
                "name": key,
                "block": stored.index(source_of(key)[0]),
                "rotate": rotation * source_of(key)[1],
                "turns": turns[key],
                "limit": AUXILIARY_LIMITS.get(key, 1800.0),
            }
            for key in auxiliary
        ],
        **blocks,
    }


def plasma_block(twin: Twin, grid, coarse_shape, strides) -> dict:
    """The plasma-current field on the coarsened grid, at the reference pressure."""
    equilibrium = twin.solve_profiles("standard", kinetics.HIGH_PERFORMANCE)
    beta = float(equilibrium.wout.betatotal)
    distribution = plasma_response.current_distribution(
        equilibrium,
        num_theta=PLASMA_THETA,
        num_zeta=PLASMA_ZETA,
        radial_stride=PLASMA_STRIDE,
    )
    stride_phi, stride_z, stride_r = strides
    coarse = dataclasses.replace(
        grid,
        num_r=int(coarse_shape[2]),
        num_z=int(coarse_shape[1]),
        num_phi=int(coarse_shape[0]),
        # Decimation keeps the first sample of each axis, so the maxima move in.
        r_max=grid.r_min
        + (int(coarse_shape[2]) - 1)
        * stride_r
        * (grid.r_max - grid.r_min)
        / (grid.num_r - 1),
        z_max=grid.z_min
        + (int(coarse_shape[1]) - 1)
        * stride_z
        * (grid.z_max - grid.z_min)
        / (grid.num_z - 1),
    )
    interpreter = GPU_PYTHON if Path(GPU_PYTHON).exists() else None
    print(
        f"plasma field at <beta> = {100 * beta:.3f} % from "
        f"{distribution.num_sources} current elements"
        + (" on the GPU" if interpreter else " on the CPU")
    )
    b_r, b_p, b_z = plasma_response.field_on_grid(
        distribution, coarse, interpreter=interpreter, verbose=True
    )
    shape = (coarse.num_phi, coarse.num_z, coarse.num_r)
    return {
        "beta_reference": beta,
        "net_toroidal_current_a": float(equilibrium.wout.ctor),
        "b_r": encode(b_r.reshape(shape)),
        "b_p": encode(b_p.reshape(shape)),
        "b_z": encode(b_z.reshape(shape)),
    }


def superconducting_agrees(twin: Twin) -> float:
    """Largest relative response difference of the seven main circuits between the extended and base coils files."""
    base = _common.twin()
    worst = 0.0
    for extended, plain in (
        (twin.response.b_r, base.response.b_r),
        (twin.response.b_p, base.response.b_p),
        (twin.response.b_z, base.response.b_z),
    ):
        extended = np.asarray(extended)[: base.coils.num_circuits]
        plain = np.asarray(plain)
        scale = float(np.max(np.abs(plain)))
        worst = max(worst, float(np.max(np.abs(extended - plain)) / max(scale, 1e-30)))
    return worst


class DecimatedField:
    """Vacuum field on the page's decimated grid under the page's own trilinear interpolation."""

    def __init__(self, source, stride_phi: int, stride_z: int, stride_r: int) -> None:
        self.num_field_periods = source.num_field_periods
        self.period = source.period
        self.b = source.b[:, ::stride_phi, ::stride_z, ::stride_r]
        self.num_phi, self.num_z, self.num_r = self.b.shape[1:]
        self.r_min = source.r_min
        self.z_min = source.z_min
        # Decimation keeps the first sample of each axis, so the last one moves in.
        self.r_max = source.r_min + (self.num_r - 1) * stride_r * source.dr
        self.z_max = source.z_min + (self.num_z - 1) * stride_z * source.dz
        self.dr = stride_r * source.dr
        self.dz = stride_z * source.dz
        self.dphi = self.period / self.num_phi
        # `digest` memoises into this slot, and anything keyed on the field wants a
        # decimated grid to hash apart from the full one it came from: the decimated
        # `b` and the moved-in extents both enter the hash, so they do.
        self._digest = None

    __call__ = field.VacuumField.__call__
    magnitude = field.VacuumField.magnitude
    digest = field.VacuumField.digest


def strike_resolution(twin: Twin, candidates) -> list[dict]:
    """Median strike displacement between each candidate grid and the full one, in metres."""
    from w7x_twin.magnetics import fieldlines

    state = twin.state("standard")
    full = field.VacuumField(twin.response, state.currents)
    vessel = _common.vessel()
    elements = _common.components()
    equilibrium = twin.solve(state, SCAN)
    starts, r_axis, z_axis, _ = fieldlines.fan_starts(
        full, equilibrium.wout, (1.0, 1.32), 40
    )

    def strikes_of(traced):
        section, _ = fieldlines.trace(
            traced, starts, np.full(starts.shape, z_axis), turns=60, plane_phi=0.0,
            vessel=vessel, components=elements,
        )
        return section.strikes

    reference = strikes_of(full)
    out = []
    for strides in candidates:
        placed = strikes_of(DecimatedField(full, *strides))
        both = reference.struck & placed.struck
        displacement = (
            np.hypot(
                reference.r[both] - placed.r[both], reference.z[both] - placed.z[both]
            )
            if both.any()
            else np.array([np.nan])
        )
        out.append(
            {
                "strides": list(strides),
                "lines": int(both.sum()),
                "median_displacement_m": float(np.median(displacement)),
                "worst_displacement_m": float(np.max(displacement)),
                "resolves_the_migration": bool(
                    np.median(displacement) < 0.5 * STRIKE_MIGRATION_M
                ),
            }
        )
    return out


def run_export_field() -> int:
    # One coils file for both blocks, so the page carries one geometry version. The
    # extended file's first seven circuits are the superconducting set.
    coils_file = AUXILIARY_COILS if (Path("data") / AUXILIARY_COILS).exists() else "coils.w7x"
    twin = _common.twin(coils_file=coils_file)
    grid = twin.coils.grid
    table = twin.response

    shape = (grid.num_phi, grid.num_z, grid.num_r)
    circuits = [key for key in twin.coils.circuit_keys if key in machine.CIRCUIT_ORDER]
    print(f"{twin.geometry}")
    print(f"response table {len(circuits)} superconducting circuits on {shape}")
    if coils_file != "coils.w7x":
        print(
            f"the superconducting response is unchanged by the appended windings to "
            f"{superconducting_agrees(twin):.2e} of its own amplitude"
        )

    # What each candidate coarsening resolves, against the distance the strike line
    # migrates with the plasma current. A page whose tracer cannot separate the two ends of
    # that migration carries the island chain and the transform but not the exhaust.
    scan = strike_resolution(twin, CANDIDATE_STRIDES)
    block_bytes = (
        np.asarray(table.b_r)
        .reshape((twin.coils.num_circuits, *shape))[: len(circuits)]
        .nbytes
    )
    layout = _common.Table(
        ("stride phi,Z,R", ">16s"), ("lines", "6d"), ("median [mm]", "12.1f"),
        ("worst [mm]", "11.1f"), ("payload [MB]", "13.2f"),
    )
    print()
    layout.begin()
    for row in scan:
        divisor = row["strides"][0] * row["strides"][1] * row["strides"][2]
        row["payload_mb"] = 3.0 * block_bytes / divisor / 1e6
        layout.row(
            str(tuple(row["strides"])), row["lines"],
            1e3 * row["median_displacement_m"], 1e3 * row["worst_displacement_m"],
            row["payload_mb"],
        )
    resolving = [row for row in scan if row["resolves_the_migration"]]
    chosen = (
        min(resolving, key=lambda r: r["payload_mb"])
        if resolving
        else min(scan, key=lambda r: r["median_displacement_m"])
    )
    stride_phi, stride_z, stride_r = chosen["strides"]
    print(
        f"the strike line migrates {1e3 * STRIKE_MIGRATION_M:.0f} mm across the bootstrap "
        f"current range, so the smallest grid that separates its two ends is "
        f"{tuple(chosen['strides'])}"
        + ("" if resolving else ", and none of the candidates does")
    )

    fields = {}
    for name, source in (("b_r", table.b_r), ("b_p", table.b_p), ("b_z", table.b_z)):
        block = np.asarray(source).reshape((twin.coils.num_circuits, *shape))
        fields[name] = block[: len(circuits), ::stride_phi, ::stride_z, ::stride_r]
    coarse_shape = fields["b_r"].shape[1:]
    print(f"coarsened to {coarse_shape}, {fields['b_r'].nbytes * 3 / 1e6:.2f} MB float32")

    payload = {
        "geometry": twin.geometry.as_dict(),
        "grid": {
            "r_min": grid.r_min,
            "r_max": grid.r_max,
            "num_r": int(coarse_shape[2]),
            "z_min": grid.z_min,
            "z_max": grid.z_max,
            "num_z": int(coarse_shape[1]),
            "num_phi": int(coarse_shape[0]),
            "num_field_periods": grid.num_field_periods,
        },
        "circuits": list(circuits),
        "turns": [int(value) for value in twin.coils.turns()[: len(circuits)]],
        "presets": [
            {
                "key": key,
                "label": machine.get(key).label,
                "currents": [float(value) for value in machine.get(key).currents],
            }
            for key in PRESETS
            if key in machine.MEASURED
        ],
        "b_r": encode(fields["b_r"]),
        "b_p": encode(fields["b_p"]),
        "b_z": encode(fields["b_z"]),
        "strike_resolution": {"chosen": chosen, "scan": scan},
    }

    # The vessel contour resampled onto the field's own toroidal grid, so a traced line
    # can be stopped where it reaches the wall without interpolating between cuts.
    vessel = load_vessel("data/vessel.part")
    period = 2.0 * np.pi / grid.num_field_periods
    angles = np.linspace(0.0, period, int(coarse_shape[0]), endpoint=False)
    cuts = [vessel.cut_at(float(angle)) for angle in angles]
    payload["vessel_contour"] = {
        "r": encode(np.array([cut[0] for cut in cuts])),
        "z": encode(np.array([cut[1] for cut in cuts])),
    }
    print(
        f"vessel contour {len(cuts)} cuts of {len(cuts[0][0])} points on the field grid"
    )

    payload["plasma"] = plasma_block(twin, grid, coarse_shape, chosen["strides"])
    auxiliary = auxiliary_block(twin, twin.cache_dir)
    if auxiliary is not None:
        payload["auxiliary"] = auxiliary

    FIELD_OUT.parent.mkdir(parents=True, exist_ok=True)
    FIELD_OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {FIELD_OUT} ({FIELD_OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


# -- page-error --------------------------------------------------------------------

# Page grid and tracer against the model's, four variants; writes results/archive/page_tracer_error.json.

TRACER_RECORD = Path("results/archive/page_tracer_error.json")

#: The page's own tracer resolution and fan, from artifact/twin3d.template.html.
PAGE_STEPS = 48
PAGE_WALL_EVERY = 8
PAGE_LAYER = (0.06, 1.30)
PAGE_LINES = 22
PAGE_TURNS = 42
#: The strike fan the model-versus-page displacement is measured on.
STRIKE_LAYER = (1.0, 1.32)
STRIKE_LINES = 24
STRIKE_TURNS = 60


def run_page_error() -> int:
    stored = json.loads(FIELD_OUT.read_text())
    strides = tuple(stored["strike_resolution"]["chosen"]["strides"])
    twin = _common.twin(coils_file="coils.w7x_full")
    state = twin.state("standard")
    full = VacuumField(twin.response, state.currents)
    page_grid = DecimatedField(full, *strides)
    vessel = _common.vessel()
    elements = _common.components()
    equilibrium = twin.solve(state, SCAN)
    r_lcfs, _ = diagnostics.boundary_cut(equilibrium.wout, 0.0)
    print(f"{twin.geometry}")
    print(f"shipped strides {strides}, page tracer {PAGE_STEPS} steps per period")

    variants = {
        "model": (full, 120, 1),
        "tracer": (full, PAGE_STEPS, PAGE_WALL_EVERY),
        "grid": (page_grid, 120, 1),
        "page": (page_grid, PAGE_STEPS, PAGE_WALL_EVERY),
    }
    rows = []
    half_width = float("nan")
    reference_strikes = None
    for name, (traced_field, steps, wall_every) in variants.items():
        axis_r, axis_z = fieldlines.find_axis(traced_field, steps_per_period=steps)
        half = float(r_lcfs.max()) - axis_r
        if name == "page":
            half_width = half

        starts = axis_r + np.linspace(*STRIKE_LAYER, STRIKE_LINES) * half
        section, _ = fieldlines.trace(
            traced_field, starts, np.full(starts.shape, axis_z), turns=STRIKE_TURNS,
            steps_per_period=steps, plane_phi=0.0, vessel=vessel,
            components=elements, wall_check_every=wall_every,
        )
        strikes = section.strikes

        fan = axis_r + np.linspace(*PAGE_LAYER, PAGE_LINES) * half
        wind_starts = np.concatenate([[axis_r], fan])
        fan_section, winding = fieldlines.trace(
            traced_field, wind_starts, np.full(wind_starts.shape, axis_z),
            turns=PAGE_TURNS, steps_per_period=steps, plane_phi=0.0,
            winding_reference=0,
        )
        # The scalar turn normalisation inside trace() only holds when every line lives
        # the full trace, so full survival is asserted rather than assumed.
        crossings = np.bincount(
            fan_section.line_index, minlength=len(wind_starts)
        )
        if not (crossings == PAGE_TURNS).all():
            raise SystemExit(
                f"{name}: {int((crossings < PAGE_TURNS).sum())} fan lines left the "
                f"grid, so the winding normalisation does not hold"
            )
        iota = winding[1:]
        finite = iota[np.isfinite(iota)]

        row = {
            "variant": name,
            "axis_r": float(axis_r),
            "axis_z": float(axis_z),
            "iota_axis": float(finite[0]) if finite.size else float("nan"),
            "iota_outermost": float(finite[-1]) if finite.size else float("nan"),
            "iota_lines": [float(v) for v in iota],
            "struck": int(strikes.struck.sum()),
        }
        if reference_strikes is None:
            reference_strikes = strikes
        else:
            both = reference_strikes.struck & strikes.struck
            displacement = np.hypot(
                reference_strikes.r[both] - strikes.r[both],
                reference_strikes.z[both] - strikes.z[both],
            )
            length_change = np.abs(
                strikes.connection_length_m[both]
                - reference_strikes.connection_length_m[both]
            ) / np.maximum(reference_strikes.connection_length_m[both], 1e-30)
            row.update(
                lines_in_both=int(both.sum()),
                strike_median_m=float(np.median(displacement)) if both.any() else float("nan"),
                strike_worst_m=float(np.max(displacement)) if both.any() else float("nan"),
                connection_median_change=float(np.median(length_change)) if both.any() else float("nan"),
            )
        rows.append(row)
        print(
            f"  {name:7s} axis {axis_r:.6f} m, iota outermost "
            f"{row['iota_outermost']:.6f}, {row['struck']} of {STRIKE_LINES} struck"
            + (
                f", strike median {1e3 * row['strike_median_m']:.1f} mm, "
                f"worst {1e3 * row['strike_worst_m']:.1f} mm"
                if "strike_median_m" in row else ""
            )
        )

    page = next(r for r in rows if r["variant"] == "page")
    model = next(r for r in rows if r["variant"] == "model")
    record = {
        "geometry": twin.geometry.as_dict(),
        "strides": list(strides),
        "page_steps_per_period": PAGE_STEPS,
        "page_wall_check_every": PAGE_WALL_EVERY,
        "strike_layer": list(STRIKE_LAYER),
        "strike_lines": STRIKE_LINES,
        "strike_turns": STRIKE_TURNS,
        "page_layer": list(PAGE_LAYER),
        "page_lines": PAGE_LINES,
        "page_turns": PAGE_TURNS,
        "half_width_m": float(half_width),
        "axis_gap_m": float(abs(page["axis_r"] - model["axis_r"])),
        "iota_gap": float(abs(page["iota_outermost"] - model["iota_outermost"])),
        "variants": rows,
    }
    TRACER_RECORD.parent.mkdir(parents=True, exist_ok=True)
    TRACER_RECORD.write_text(json.dumps(record, indent=2))
    print(
        f"\nagainst the model: axis {1e3 * record['axis_gap_m']:.3f} mm, winding "
        f"{record['iota_gap']:.2e}, strike median "
        f"{1e3 * page.get('strike_median_m', float('nan')):.1f} mm"
    )
    print(f"wrote {TRACER_RECORD}")
    return 0


# -- records -----------------------------------------------------------------------

# Audit every computed record against the geometry and the inputs it names.
#
#     python -m w7x_twin records

#: Export bundles the page and the renderer consume. They carry a geometry stamp of
#: their own but are megabytes of interpolation table, so the audit names them and
#: does not open them.
EXPORT_BUNDLES = (
    "results/magnetics/w7x_field.json",
    "results/magnetics/w7x_geometry.json",
    "results/magnetics/w7x_machine_meshes.json",
)

#: Records that carry no geometry stamp, and why they carry none.
UNSTAMPED: dict[str, str] = {
    "results/hardware/toolchain.json":
        "a roster of external binaries, which no geometry enters",
    "results/turbulence/shear_quench.json":
        "no command in this package writes it, so nothing stamps it",
}

#: The command that writes each record, so a stale entry names what refreshes it.
WRITTEN_BY: dict[str, str] = {
    "results/archive/page_tracer_error.json": "page-error",
    "results/benchmarks/koeberl.json": "koeberl",
    "results/discharges/discharge_history.json": "history",
    "results/discharges/intrinsic_error_field.json": "intrinsic",
    "results/discharges/profile_residuals.json": "profiles",
    "results/discharges/reproduce_discharge.json": "discharge",
    "results/discharges/symmetrise.json": "symmetrise",
    "results/discharges/trim_radius.json": "trim-radius",
    "results/equilibrium/beta_scan_standard.json": "beta standard",
    "results/equilibrium/config_survey.json": "equilibrium",
    "results/equilibrium/ensemble.json": "ensemble",
    "results/equilibrium/island_equilibrium.json": "stepped",
    "results/equilibrium/spec.json": "spec",
    "results/equilibrium/stability_limits.json": "stability",
    "results/equilibrium/winding_pack.json": "winding",
    "results/exhaust/heat_flux.json": "exhaust",
    "results/exhaust/recycling_pressure.json": "recycling",
    "results/exhaust/strike_line_migration.json": "migration",
    "results/exhaust/target_incidence.json": "incidence",
    "results/hardware/cad_geometry.json": "cad",
    "results/magnetics/error_field.json": "errorfield",
    "results/magnetics/plasma_response.json": "response",
    "results/plasma/ambipolar_field.json": "efield",
    "results/plasma/bootstrap_routes.json": "bootstrap",
    "results/plasma/computed.json": "computed",
    "results/plasma/coupled_solve.json": "coupled",
    "results/plasma/deposition.json": "deposition",
    "results/plasma/solved_density.json": "density",
    "results/plasma/transient_discharge.json": "transient",
    "results/plasma/turbulent_transport.json": "turbulence",
    "results/turbulence/growth_rate_grid.json": "growth-rate-grid",
    "results/turbulence/gyrokinetic.json": "gyrokinetic",
    "results/turbulence/mixing_length_constant.json": "saturation",
    "results/validation.json": "validate",
}

#: Records whose commands stamp the inputs they read. Without this, a record carrying
#: no ``reads`` block cannot be told from one that reads nothing.
DECLARES_READS: dict[str, tuple[str, ...]] = {
    "results/discharges/profile_residuals.json": (
        "results/turbulence/growth_rate_grid.json",
        "results/turbulence/mixing_length_constant.json",
    ),
    "results/exhaust/recycling_pressure.json": ("results/exhaust/heat_flux.json",),
    "results/plasma/computed.json": (
        "results/turbulence/growth_rate_grid.json",
        "results/turbulence/mixing_length_constant.json",
    ),
    "results/plasma/transient_discharge.json": ("results/exhaust/heat_flux.json",),
    "results/turbulence/mixing_length_constant.json": (
        "results/turbulence/growth_rate_grid.json",
    ),
    "results/plasma/turbulent_transport.json": (
        "results/turbulence/growth_rate_grid.json",
        "results/turbulence/mixing_length_constant.json",
    ),
}

#: A record names the coils file its analysis chose, so a difference in these parts is
#: deliberate. A difference in any other part means the input was rewritten after the
#: record was written.
CHOSEN_PARTS = ("coils", "constructed")


current_geometry = _common.current_geometry


def audit_record(path: Path, reference) -> dict:
    """One record against the current geometry and against the inputs it names."""
    name = path.as_posix()
    entry: dict = {"record": name, "stale_parts": [], "stale_reads": [], "notes": []}
    try:
        stored = json.loads(path.read_text())
    except ValueError as error:
        entry["notes"].append(f"unreadable: {error}")
        return entry

    geometry = stored.get("geometry") if isinstance(stored, dict) else None
    if not isinstance(geometry, dict):
        if name in UNSTAMPED:
            entry["notes"].append(UNSTAMPED[name])
        else:
            entry["stale_parts"].append("no geometry stamp")
    else:
        # An epoch is part of the geometry, so a record is compared against the
        # version of its own epoch.
        era = geometry.get("epoch", machine.DEFAULT_EPOCH)
        expected = dict(
            reference.parts if era == machine.DEFAULT_EPOCH
            else current_geometry(era).parts
        )
        for part, value in expected.items():
            held = geometry.get(part)
            if held is None:
                entry["stale_parts"].append(f"{part} absent")
            elif held != value and part not in CHOSEN_PARTS:
                entry["stale_parts"].append(f"{part} {held} against {value}")

    reads = stored.get("reads") if isinstance(stored, dict) else None
    declared = DECLARES_READS.get(name)
    if declared and not isinstance(reads, dict):
        entry["stale_reads"].append(
            "no input digests, though the command stamps "
            + ", ".join(Path(source).name for source in declared)
        )
    elif isinstance(reads, dict):
        for source, digest in reads.items():
            now = _common.file_digest(source)
            if now != digest:
                entry["stale_reads"].append(f"{Path(source).name} {digest} against {now}")
    return entry


def run_records() -> int:
    """Report which records were computed against inputs that have since moved."""
    if not (Path("data") / "coils.w7x").is_file():
        raise SystemExit("no data/coils.w7x; run python -m w7x_twin fetch")
    reference = current_geometry()
    print(f"current {reference}")
    print()

    paths = sorted(
        path for path in Path("results").rglob("*.json")
        if path.as_posix() not in EXPORT_BUNDLES
    )
    rows = [audit_record(path, reference) for path in paths]

    layout = _common.Table(
        ("record", "48s"), ("geometry", ">10s"), ("inputs", ">8s"), ("refresh with", "s")
    )
    layout.begin(extra=20)
    for row in rows:
        command = WRITTEN_BY.get(row["record"], "")
        layout.row(
            row["record"].removeprefix("results/"),
            "stale" if row["stale_parts"] else ("-" if row["notes"] else "ok"),
            "stale" if row["stale_reads"] else (
                "ok" if row["record"] in DECLARES_READS else "-"
            ),
            f"w7x-twin {command}" if command else "",
        )

    print()
    stale = [row for row in rows if row["stale_parts"] or row["stale_reads"]]
    for row in stale:
        print(f"{row['record']}")
        for reason in row["stale_parts"] + row["stale_reads"]:
            print(f"    {reason}")
    for row in rows:
        for note in row["notes"]:
            print(f"{row['record']}\n    {note}")

    print()
    print(
        f"{len(rows) - len(stale)} of {len(rows)} records stand on the current geometry "
        f"and the current inputs"
    )
    return 1 if stale else 0


# -- validate ----------------------------------------------------------------------

# Validate against published values and an independent coil set; non-zero exit on any disagreement.

MU0 = 4.0e-7 * np.pi
RECORD = Path("results/validation.json")

_records: list[dict] = []
_section = ""


def check(
    name: str,
    value: float,
    published: str,
    low: float,
    high: float,
    unit: str = "",
) -> None:
    """Compare one computed quantity against the band its reference allows."""
    agrees = bool(low <= value <= high)
    mark = "ok  " if agrees else "??  "
    print(
        f"  {mark}{name:38s} {value:12.4f} {unit:6s}  published: {published}"
        f"   [{low:g}, {high:g}]"
    )
    _records.append(
        {
            "section": _section,
            "quantity": name,
            "value": float(value),
            "unit": unit,
            "published": published,
            "band": [float(low), float(high)],
            "agrees": agrees,
        }
    )


def section(title: str) -> None:
    global _section
    _section = title
    print(f"\n{title}\n" + "-" * len(title))


def write_record(geometry) -> int:
    disagreed = [record for record in _records if not record["agrees"]]
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(
        json.dumps(
            {
                "geometry": geometry.as_dict(),
                "checks": _records,
                "agreed": len(_records) - len(disagreed),
                "total": len(_records),
            },
            indent=2,
        )
    )
    print(f"\n{len(_records) - len(disagreed)} of {len(_records)} checks agree")
    for record in disagreed:
        print(f"  disagrees: {record['section']} / {record['quantity']}")
    print(f"wrote {RECORD}")
    return 1 if disagreed else 0


def compare_with_simsopt(coils: machine.CoilSet) -> None:
    """Filament geometry against simsopt's independently derived W7-X coil set."""
    try:
        from simsopt.configs import get_w7x_data
    except Exception as error:  # pragma: no cover - optional dependency
        print(f"  simsopt unavailable ({error})")
        return

    curves, _currents, _ = get_w7x_data()
    ours = [group[0] for group in coils.filaments]
    theirs = [np.asarray(c.gamma()) for c in curves[: len(ours)]]

    for key, mine, other in zip(coils.circuit_keys, ours, theirs, strict=False):
        length_mine = float(np.sum(np.linalg.norm(np.diff(mine, axis=0), axis=1)))
        closed = np.vstack([other, other[:1]])
        length_other = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
        difference = 100.0 * abs(length_mine - length_other) / length_other
        check(
            f"filament length, {key}, against simsopt",
            difference,
            f"{length_other:.4f} m",
            0.0,
            0.05,
            "%",
        )


def coil_current_for_field(twin: Twin, target_t: float = 2.5) -> float:
    """Modular current per turn giving a volume-averaged axis field of ``target_t``."""
    lo, hi = 10000.0, 18000.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        output = twin.solve(twin.state("standard", field_scale=mid / 13000.0), SCAN)
        b0 = float(output.wout.b0)
        if abs(b0 - target_t) < 1e-4:
            return mid
        if b0 < target_t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_validate() -> int:
    twin = _common.twin()
    coils = twin.coils

    section("Geometry")
    print(f"  {twin.geometry}")
    print(f"  equilibrium keys on coils and grid: {twin.geometry.subset('coils', 'grid')}")

    section("Coil geometry against simsopt's independent W7-X set")
    compare_with_simsopt(coils)

    section("Vacuum field, standard configuration at 12883 A/turn")
    currents = np.array([12883.0] * 5 + [0.0, 0.0])
    scan = field.field_on_axis_scan(twin.response, currents, 5.931, 144)
    check("|B| at R=5.931 m, bean plane", scan[0], "2.5 T nominal", 2.50, 2.54, "T")
    mirror = 100.0 * (scan.max() - scan.min()) / (scan.max() + scan.min())
    check("mirror term on that circle", mirror, "about 5 %", 4.4, 4.9, "%")

    section("Equilibrium, standard configuration")
    standard = twin.solve(twin.state("standard"), SCAN)
    d = diagnostics.analyse(standard)
    check("major radius", d.major_radius_m, "5.5 m", 5.49, 5.54, "m")
    check("minor radius", d.minor_radius_m, "about 0.5 m", 0.48, 0.50, "m")
    check("aspect ratio", d.aspect_ratio, "about 11", 11.0, 11.5)
    check("plasma volume", d.plasma_volume_m3, "26-30 m^3", 25.8, 26.7, "m^3")
    check("transform on axis", d.iota_axis, "about 0.85", 0.840, 0.860)
    check("transform at edge", d.iota_edge, "just below 5/5", 0.945, 0.965)
    check("mirror term", d.mirror_percent, "about 5 %", 4.25, 4.55, "%")
    check("magnetic well", 100 * d.magnetic_well_depth, "about 1 %", 0.90, 1.15, "%")

    section("Equilibrium, high mirror (IPP reference 167)")
    high = twin.solve(twin.state("high_mirror_ref167"), SCAN)
    dh = diagnostics.analyse(high)
    check("mirror term", dh.mirror_percent, "about 10 %", 9.8, 10.4, "%")
    check("transform on axis", dh.iota_axis, "about 0.86", 0.850, 0.865)

    section("Operating point")
    current = coil_current_for_field(twin, 2.5)
    check(
        "modular current for B0 = 2.5 T",
        current,
        f"below the {machine.MAX_COIL_CURRENT_A:.0f} A conductor limit",
        13700.0,
        13850.0,
        "A/turn",
    )
    check(
        "amp-turns per coil",
        current * 108 / 1e6,
        "about 1.5 MA-turns",
        1.47,
        1.51,
        "MA",
    )

    section("Traced field, sharing no machinery with the equilibrium solver")
    from w7x_twin.magnetics import fieldlines
    from w7x_twin.magnetics.field import VacuumField

    vacuum = VacuumField(twin.response, currents)
    axis_r, axis_z = fieldlines.find_axis(vacuum)
    check("traced magnetic axis", axis_r, "R = 5.947 m", 5.94, 5.955, "m")
    check("|B| on the traced axis", float(vacuum.magnitude(axis_r, 0.0, axis_z)[0]),
          "2.5 T", 2.49, 2.52, "T")

    section("Records the newer analyses wrote, checked where they stand")
    stored_checks(Path("results"))

    section("Record dependencies")
    dependency_checks(Path("results"))

    return write_record(twin.geometry)


def dependency_checks(results: Path) -> None:
    """Records carrying input digests re-verified, so a stale dependency fails validation."""
    stamped = 0
    for record_path in sorted(results.rglob("*.json")):
        try:
            stored = json.loads(record_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        reads = stored.get("reads") if isinstance(stored, dict) else None
        if not isinstance(reads, dict) or not reads:
            continue
        stamped += 1
        stale = sum(
            1 for source, digest in reads.items()
            if _common.file_digest(source) != digest
        )
        check(
            f"{record_path.relative_to(results)} inputs current",
            float(stale),
            f"digests of the {len(reads)} records it read",
            0.0, 0.0,
        )
    if not stamped:
        print("  no record carries input digests yet")


def stored_checks(results: Path) -> None:
    """Written records re-asserted, so a regression fails validation like a wrong solve."""
    trim = results / "discharges/trim_radius.json"
    if trim.exists():
        stored = json.loads(trim.read_text())
        band = stored.get("pinned_radius_band_m") or [float("nan")] * 2
        check(
            "trim mounting radius, pinned",
            stored.get("pinned_radius_m", float("nan")),
            "outer vessel outboard surface at 7.570 m",
            band[0], band[1], "m",
        )

    geometry = results / "hardware/cad_geometry.json"
    if geometry.exists():
        stored = json.loads(geometry.read_text())
        packs = stored.get("winding_packs", {})
        if packs:
            check(
                "winding pack, turn direction",
                packs.get("non_planar_turn_direction_mm", float("nan")),
                "156.0 mm at the published pitch",
                153.5, 158.0, "mm",
            )
        if "non_planar_layer_direction_mm" in packs:
            check(
                "winding pack, layer direction",
                packs.get("non_planar_layer_direction_mm", float("nan")),
                "inside the 208.5 to 225.4 mm the stagger left",
                208.5, 225.4, "mm",
            )
        vessel = stored.get("outer_vessel", {}).get("outboard_radius_m", {})
        if vessel:
            check(
                "outer vessel outboard radius",
                vessel.get("median", float("nan")),
                "inside the pinned 7.000 to 7.690 m",
                7.000, 7.690, "m",
            )
        boundary = stored.get("plasma_boundary", {}).get("residual_mm", {})
        if boundary:
            check(
                "released plasma model, median residual",
                boundary.get("median", float("nan")),
                "the released surface against the twin's boundary",
                0.0, 35.0, "mm",
            )
        centre = stored.get("coil_centrelines", {})
        if centre:
            check(
                "filaments inside the released coil solids",
                centre.get("worst_median_mm", float("nan")),
                "winding-package asymmetry scale",
                0.0, 20.0, "mm",
            )
        part = stored.get("vessel_part", {})
        if part:
            check(
                "vessel.part against the released vessel",
                part.get("median_mm", float("nan")),
                "the contour every trace stops on",
                0.0, 25.0, "mm",
            )

    spec = results / "equilibrium/spec.json"
    if spec.exists():
        stored = json.loads(spec.read_text())
        rows = [r for r in stored.get("cases", []) if np.isfinite(r.get("force_residual", float("nan")))]
        bracketing = [r for r in rows if r.get("placement") == "bracketing it"
                      and np.isfinite(r.get("island_mm", float("nan"))) and r.get("island_mm", 0) > 0]
        if bracketing:
            # The island is measured on the converged solve: a stalled case's section
            # is not force balance, so its island is not the record's answer.
            converged = min(bracketing, key=lambda r: r["force_residual"])
            check(
                "stepped-pressure island, interfaces bracketing",
                converged["island_mm"],
                "an island the placement leaves room for, on the best-converged case",
                5.0, 200.0, "mm",
            )

    intrinsic = results / "discharges/intrinsic_error_field.json"
    if intrinsic.exists():
        stored = json.loads(intrinsic.read_text())
        summary = stored.get("summary", [])
        if summary:
            ratios = [s["b22_against_published"] for s in summary]
            check(
                "coil deviations separated by the second harmonic",
                max(ratios) / max(min(ratios), 1e-9),
                "a discrimination the 1/1 alone cannot make",
                10.0, 1e6, "x",
            )
