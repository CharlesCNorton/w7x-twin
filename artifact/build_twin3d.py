"""Assemble the 3D twin as one self-contained page.

Inlines the renderer library, the two typefaces and the exported geometry so the page
depends on nothing it does not carry. The library ships as an ES module whose minified
build renames its internals, so its trailing export statement is rewritten into an
object and the names the page uses are taken from it. The faces are fetched once into a
sibling cache and embedded as data URIs: Barlow Semi Condensed sets the labels, IBM Plex
Mono every figure, Latin subset only, both under the SIL Open Font License.

    build_twin3d.py [geometry.json] [output.html]
"""

from __future__ import annotations

import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
THREE_URL = "https://unpkg.com/three@0.160.0/build/three.module.min.js"
CACHE = HERE / "_three.module.min.js"
FONTS = HERE / "_fonts.css"

#: Names the page takes from the library.
USED = (
    "WebGLRenderer Scene Color FogExp2 PerspectiveCamera AmbientLight DirectionalLight "
    "Plane Vector3 Group BufferGeometry BufferAttribute Line LineBasicMaterial Mesh "
    "MeshStandardMaterial MeshPhysicalMaterial DoubleSide BackSide Points PointsMaterial "
    "Spherical MathUtils Box3 Sphere"
).split()

FAMILIES = (("Barlow+Semi+Condensed", (500, 600)), ("IBM+Plex+Mono", (400,)))
AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}
#: The Latin block, which identifies the subset to keep.
LATIN_MARKER = "U+0000-00FF"


def fetch(url: str, headers: dict | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        return response.read()


def library_source() -> str:
    """The library with its export statement rewritten into a plain object."""
    if not CACHE.exists():
        CACHE.write_bytes(fetch(THREE_URL))
        print(f"fetched {THREE_URL} ({CACHE.stat().st_size / 1024:.0f} kB)")
    source = CACHE.read_text(encoding="utf-8")

    match = None
    for match in re.finditer(r"export\s*\{", source):
        pass
    if match is None:
        raise SystemExit("no export statement found in the library")

    start = match.end()
    end = source.index("}", start)
    entries = []
    for item in source[start:end].split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(" as ")
        internal = parts[0].strip()
        public = parts[-1].strip()
        entries.append(f"{public}:{internal}")

    rewritten = (
        source[: match.start()]
        + "const THREE={"
        + ",".join(entries)
        + "};\n"
        + source[end + 1 :].lstrip().lstrip(";")
    )
    missing = [name for name in USED if f"{name}:" not in rewritten]
    if missing:
        raise SystemExit(f"library does not export {missing}")
    return rewritten + f"\nconst {{{','.join(USED)}}} = THREE;\n"


def fonts_css() -> str:
    """The two typefaces as data URIs, fetched into the sibling cache when absent."""
    if not FONTS.exists():
        blocks: list[str] = []
        for family, weights in FAMILIES:
            specification = "wght@" + ";".join(str(weight) for weight in weights)
            stylesheet = fetch(
                f"https://fonts.googleapis.com/css2?family={family}:{specification}&display=swap",
                AGENT,
            ).decode()
            for body in re.findall(r"@font-face\s*\{([^}]*)\}", stylesheet):
                if LATIN_MARKER not in body:
                    continue
                name = re.search(r"font-family: '([^']+)'", body).group(1)
                weight = int(re.search(r"font-weight:\s*(\d+)", body).group(1))
                url = re.search(r"url\((https://[^)]+\.woff2)\)", body).group(1)
                payload = fetch(url)
                print(f"  {name:24s} {weight}  {len(payload) / 1024:5.1f} kB")
                blocks.append(
                    f"@font-face{{font-family:'{name}';font-style:normal;"
                    f"font-weight:{weight};font-display:block;"
                    f"src:url(data:font/woff2;base64,{base64.b64encode(payload).decode()}) "
                    "format('woff2');}"
                )
        FONTS.write_text("\n".join(blocks), encoding="utf-8")
        print(f"wrote {FONTS} ({FONTS.stat().st_size / 1024:.0f} kB)")
    return FONTS.read_text(encoding="utf-8")


#: Which command produced a stored verification row, by its quantity prefix; the live
#: rows come from the validate command itself.
READOUT_COMMANDS = (
    ("trim mounting radius", "trim-radius"),
    ("winding pack", "cad"),
    ("outer vessel", "cad"),
    ("released plasma model", "cad"),
    ("filaments inside", "cad"),
    ("vessel.part", "cad"),
    ("stepped-pressure island", "spec"),
    ("coil deviations", "intrinsic"),
)


def readout_bundle(path: Path) -> str:
    """The verification record as the page's readout: value, band and command."""
    record = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for check in record["checks"]:
        command = "validate"
        for prefix, name in READOUT_COMMANDS:
            if check["quantity"].startswith(prefix):
                command = name
                break
        rows.append(
            {
                "section": check["section"],
                "quantity": check["quantity"],
                "value": check["value"],
                "unit": check["unit"],
                "low": check["band"][0],
                "high": check["band"][1],
                "agrees": check["agrees"],
                "command": command,
            }
        )
    return json.dumps(
        {
            "geometry": record["geometry"],
            "agreed": record["agreed"],
            "total": record["total"],
            "rows": rows,
        }
    )


def main() -> int:
    geometry_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/magnetics/w7x_geometry.json")
    field_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/magnetics/w7x_field.json")
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "w7x_twin3d.html"
    validation_path = Path("results/validation.json")
    for path, script in (
        (geometry_path, "python -m w7x_twin export-geometry"),
        (field_path, "python -m w7x_twin export-field"),
        (validation_path, "python -m w7x_twin validate"),
    ):
        if not path.exists():
            raise SystemExit(f"no bundle at {path}; run {script}")

    template = (HERE / "twin3d.template.html").read_text(encoding="utf-8")
    page = template.replace("{{FONTS_CSS}}", fonts_css())
    page = page.replace("{{THREE_JS}}", library_source())
    page = page.replace("{{GEOMETRY_JSON}}", geometry_path.read_text(encoding="utf-8"))
    page = page.replace("{{FIELD_JSON}}", field_path.read_text(encoding="utf-8"))
    page = page.replace("{{READOUT_JSON}}", readout_bundle(validation_path))
    output.write_text(page, encoding="utf-8")
    print(
        f"wrote {output} ({output.stat().st_size / 1024 / 1024:.2f} MB) from "
        f"{geometry_path.name} ({geometry_path.stat().st_size / 1024 / 1024:.2f} MB), "
        f"{field_path.name} ({field_path.stat().st_size / 1024 / 1024:.2f} MB) and "
        f"{validation_path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
