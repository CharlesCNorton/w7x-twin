"""Tessellate the released CAD into the mesh buffers the rendered page loads.

The release is trimmed NURBS, so a geometry kernel does the meshing: gmsh carries
OpenCascade and reads the STEP directly. Each component becomes one block of float32
vertices and uint32 triangles, base64 in a single manifest, with the source file's SHA256
so the page's machine layer carries the same provenance as every other input.

    python tools/tessellate_cad.py [component ...]
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("data/cad")
OUT = Path("results/magnetics/w7x_machine_meshes.json")

#: Component name -> (file, target element size in mm). The size is chosen per part: the
#: exhaust surfaces carry the strike lines and stay finer than the shells around them.
COMPONENTS: dict[str, tuple[str, float]] = {
    "plasma": ("01-W7-X-plasma/01-W7-X plasma--CATPRT-088633-r001.stp", 120.0),
    "divertor": ("02-W7-X-divertor/02-W7-X divertor--CATPRT-088631-r001_ultra-zip.stp", 60.0),
    "baffle": ("03-W7-X-baffle-and-divertor-closure/03-W7-X baffle and divertor closure--CATPRT-088632-r001.stp", 80.0),
    "heat_shield": ("04-W7-X-heat-shield/04-W7-X heat shield--CATPRT-088630-r001.stp", 100.0),
    "plasma_vessel": ("05-W7-X-plasma-vessel/05-W7-X plasma vessel--CATPRT-088629-r001.stp", 150.0),
    "non_planar_coils": ("06-w7-x_non-planar_coils/06-W7-X non-planar coils--CATPRT-088628-r001.stp", 100.0),
    "planar_coils": ("07-w7-x_planar_coils/07-W7-X planar coils--CATPRT-088627-r001.stp", 100.0),
    "outer_vessel": ("12-w7-x_outer_vessel/12-W7-X outer vessel--CATPRT-088621-r001.stp", 250.0),
}


def resolve(name: str) -> Path:
    stem, _ = COMPONENTS[name]
    path = ROOT / stem
    if path.exists():
        return path
    # The divertor archive unpacks under a name with the compression suffix stripped.
    matches = list((ROOT / stem.split("/")[0]).glob("*.stp"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(stem)


def tessellate(
    path: Path, size_mm: float, no_heal: bool = False, single_thread: bool = False
) -> tuple[np.ndarray, np.ndarray, str]:
    """One STEP file as welded vertices in metres, uint32 triangles, and a note.

    A release this size carries faces the kernel's healing refuses, so the import is
    retried without healing, the meshing failure of one face is not the failure of the
    file, and whatever meshed is harvested with the note saying so.
    """
    import gmsh

    note = ""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1 if single_thread else 8)
        if no_heal:
            gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
            gmsh.option.setNumber("Geometry.OCCSewFaces", 0)
        try:
            gmsh.open(str(path))
        except Exception:
            gmsh.clear()
            gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
            gmsh.option.setNumber("Geometry.OCCSewFaces", 0)
            gmsh.open(str(path))
            note = "imported without healing"
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_mm / 5.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 16)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as failure:
            note = (note + "; " if note else "") + f"partial mesh: {failure}"
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        _, connectivity = gmsh.model.mesh.getElementsByType(2)
        if not len(connectivity):
            raise RuntimeError(f"nothing meshed: {note or 'no triangles returned'}")
        vertices = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3) / 1e3
        order = {int(tag): index for index, tag in enumerate(tags)}
        triangles = np.array(
            [order[int(node)] for node in connectivity], dtype=np.uint32
        ).reshape(-1, 3)
    finally:
        gmsh.finalize()
    return vertices.astype(np.float32), triangles, note


def main() -> int:
    wanted = sys.argv[1:] or list(COMPONENTS)
    failed: list[str] = []
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Each component runs in its own process, so a kernel that dies in native code costs
    # that component and not the batch. The child is this same script with one name.
    if len(wanted) > 1 and not os.environ.get("TESSELLATE_CHILD"):
        import subprocess

        for name in wanted:
            completed = subprocess.run(
                [sys.executable, __file__, name],
                env={**os.environ, "TESSELLATE_CHILD": "1"},
            )
            if completed.returncode != 0:
                print(f"{name}: child exited {completed.returncode}")
                failed.append(name)
        if failed:
            print("failed: " + ", ".join(failed))
        return 1 if failed else 0

    manifest = json.loads(OUT.read_text()) if OUT.exists() else {"components": {}}

    for name in wanted:
        path = resolve(name)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        stored = manifest["components"].get(name)
        if stored and stored.get("source_sha256") == digest.hexdigest():
            print(f"{name}: unchanged, {stored['triangles']} triangles kept")
            continue

        started = time.monotonic()
        print(f"{name}: meshing {path.name} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
        forced = name in os.environ.get("TESSELLATE_LOOPS", "").split(",")
        try:
            if forced:
                raise RuntimeError("face-loop triangulation forced")
            vertices, triangles, note = tessellate(
                path, COMPONENTS[name][1],
                no_heal=bool(os.environ.get("TESSELLATE_NO_HEAL")),
                single_thread=bool(os.environ.get("TESSELLATE_SINGLE_THREAD")),
            )
        except Exception as failure:
            # The kernel refused or was told to stand aside; the face loops carry the
            # outline of every face, so the fallback triangulates from the file itself.
            sys.path.insert(0, "src")
            from w7x_twin.hardware import cad as cad_reader

            try:
                vertices, triangles, counts = cad_reader.planar_face_triangles(path)
            except Exception as second:
                print(f"{name}: failed, {failure}; fallback failed, {second}", flush=True)
                failed.append(name)
                continue
            note = (
                f"face-loop triangulation ({counts['planar']} planar, "
                f"{counts['chorded']} chorded faces)"
                + ("" if forced else f" after: {failure}")
            )
        # Reload before merging: another component's process may have written since this
        # one loaded, and a stale in-memory copy silently drops its work.
        manifest = json.loads(OUT.read_text()) if OUT.exists() else {"components": {}}
        manifest["components"][name] = {
            "source": path.name,
            "source_sha256": digest.hexdigest(),
            "element_size_mm": COMPONENTS[name][1],
            "vertices": int(len(vertices)),
            "triangles": int(len(triangles)),
            "note": note,
            "positions_b64": base64.b64encode(vertices.tobytes()).decode(),
            "indices_b64": base64.b64encode(triangles.tobytes()).decode(),
        }
        OUT.write_text(json.dumps(manifest))
        print(
            f"{name}: {len(vertices)} vertices, {len(triangles)} triangles in "
            f"{time.monotonic() - started:.0f} s"
            + (f" ({note})" if note else "")
            + f"; manifest {OUT.stat().st_size / 1e6:.1f} MB",
            flush=True,
        )
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
