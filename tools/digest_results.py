"""Reduce the computed records to one document a reader can hold at once.

`results/` is 148 MB, of which 144 MB is three export bundles the rendered page
consumes and nothing reads as physics. What remains is 1 MB across 36 records,
and most of that is traces and per-bin profiles: 601 time points here, 1800
gyrokinetic runs there, none of which a reader needs in full to see what a
record says.

This walks every record and writes `results/DIGEST.md`: the provenance stamp,
every scalar verbatim, and for each array its length, range, median and end
points in place of its elements. Arrays are summarised and never fitted, so
what is dropped is visible rather than smoothed away; the record itself is
named beside each entry for the cases where the elements are what is wanted.

    python tools/digest_results.py [--out results/DIGEST.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

#: Directories of records the digest covers: what the analyses computed, and the
#: measurement inputs digitised from the published figures.
ROOTS = ("results", "src/w7x_twin/records")

#: Export bundles the page consumes. Their size is reported and their contents
#: are never opened: the field table is a 121 x 121 x 36 interpolation grid per
#: circuit and the mesh file a tessellation, neither of which carries a number a
#: reader states.
BUNDLES: dict[str, str] = {
    "results/magnetics/w7x_field.json":
        "per-circuit vacuum field response for the rendered page, "
        "61 x 61 x 36 per period after coarsening; written by `export-field`",
    "results/magnetics/w7x_geometry.json":
        "coils, vessel, components and traced lines for the rendered page; "
        "written by `export-geometry`",
    "results/magnetics/w7x_machine_meshes.json":
        "CAD solids as mesh buffers; written by `tools/tessellate_cad.py`, "
        "untracked",
}

#: Arrays shorter than this are printed in full; longer ones are summarised.
INLINE_ARRAY = 8
#: Lists of records shorter than this are printed row by row; longer ones are
#: reduced to a column census, which is the only place a scalar of a record does
#: not reach this document. One table is over it: the 1800-run growth-rate grid.
INLINE_ROWS = 80
#: Distinct values of a string column reported by name rather than by count.
INLINE_LEVELS = 8
#: Characters of a string value carried before it is cut. The records carry their
#: own prose, and a source or a note repeated once per entity is the same
#: sentence many times over.
MAX_TEXT = 180
#: Entities a dict must hold before it is read as a table rather than flattened.
#: A record keyed by configuration, element or drive carries the same fields
#: under each key, which is a table written as nested dictionaries.
MIN_UNIFORM = 3


#: Home directories in a string value, whichever platform wrote it. A record can
#: carry the path a tool was run from, and this document is committed, so the
#: account and machine behind a path do not travel with it.
HOME_PATHS = (
    re.compile(r"/home/[^/\s\"']+"),
    re.compile(r"/Users/[^/\s\"']+"),
    re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\\\s\"']+"),
)


def redact(value: str) -> str:
    """A string with any home directory reduced to a placeholder."""
    for pattern in HOME_PATHS:
        value = pattern.sub("~", value)
    return value


def text(value: str) -> str:
    """A string value, cut where the records carry prose rather than a label."""
    value = redact(" ".join(str(value).split()))
    if len(value) <= MAX_TEXT:
        return value
    return f"{value[:MAX_TEXT]}... ({len(value)} chars)"


def number(value) -> str:
    """A float at the precision a reader compares against a published value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    return f"{value:.6g}"


def is_numeric(values) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def finite(values) -> list[float]:
    return [
        float(v) for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def trend(values: list[float]) -> str:
    """Whether a series rises, falls, or turns, which is what a trace is read for."""
    if len(values) < 3:
        return ""
    rises = sum(1 for a, b in zip(values, values[1:]) if b > a)
    falls = sum(1 for a, b in zip(values, values[1:]) if b < a)
    steps = max(rises + falls, 1)
    if rises / steps > 0.98:
        return "rising"
    if falls / steps > 0.98:
        return "falling"
    peak = values.index(max(values))
    if 0 < peak < len(values) - 1:
        return f"peaks at index {peak}"
    return "not monotone"


def settling(values: list[float], tolerance: float = 0.02) -> str:
    """Index from which the series stays within ``tolerance`` of its final value."""
    if len(values) < 4:
        return ""
    final = values[-1]
    scale = max(abs(final), 1e-30)
    index = len(values) - 1
    while index > 0 and abs(values[index - 1] - final) / scale <= tolerance:
        index -= 1
    if index in (0, len(values) - 1):
        return ""
    return f"within 2 % of its last value from index {index}"


def array_line(path: str, values: list) -> str:
    """One array as its length, range, median and end points."""
    count = len(values)
    if not is_numeric(values):
        sample = ", ".join(str(v) for v in values[:4])
        return f"- `{path}` [{count}] {sample}{', ...' if count > 4 else ''}"
    usable = finite(values)
    if not usable:
        return f"- `{path}` [{count}] every entry non-finite"
    parts = [
        f"[{count}]",
        f"{number(min(usable))} to {number(max(usable))}",
        f"median {number(median(usable))}",
        f"ends {number(values[0])} -> {number(values[-1])}",
    ]
    for note in (trend(usable), settling(usable)):
        if note:
            parts.append(note)
    if len(usable) != count:
        parts.append(f"{count - len(usable)} non-finite")
    return f"- `{path}` " + ", ".join(parts)


def column_census(rows: list[dict], path: str) -> list[str]:
    """Per-column range of a run table, in place of its rows."""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    out = [
        f"- `{path}` {len(rows)} rows, per-row values not carried here; "
        f"columns:"
    ]
    for key in keys:
        values = [row.get(key) for row in rows if key in row]
        usable = finite(values)
        if usable and len(usable) > len(values) / 2:
            out.append(
                f"    - `{key}` {number(min(usable))} to {number(max(usable))}, "
                f"median {number(median(usable))}"
                + (f", {len(values) - len(usable)} non-finite" if len(usable) != len(values) else "")
            )
            continue
        levels = []
        for value in values:
            label = text(value)
            if label not in levels:
                levels.append(label)
        if len(levels) <= INLINE_LEVELS:
            out.append(f"    - `{key}` " + ", ".join(levels))
        else:
            out.append(f"    - `{key}` {len(levels)} distinct values")
    return out


def cell(value) -> str:
    """One field of a table row, arrays reduced to their extent."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return number(value)
    if isinstance(value, list):
        if value and is_numeric(value):
            usable = finite(value)
            if len(value) <= 4:
                return "[" + ", ".join(number(v) for v in value) + "]"
            if usable:
                return f"[{len(value)}: {number(min(usable))} to {number(max(usable))}]"
        return f"[{len(value)}]"
    if isinstance(value, dict):
        return "{" + ", ".join(value) + "}"
    return text(value)


def nested(value, path: str) -> list[str]:
    """A field that is itself structured, indented under the row carrying it."""
    inner: list[str] = []
    walk(value, path, inner)
    return ["    " + line for line in inner]


def flat(value) -> bool:
    """True where a field fits on its row rather than needing its own walk."""
    if isinstance(value, dict):
        return False
    if isinstance(value, list):
        return not any(isinstance(item, (list, dict)) for item in value)
    return True


def row_cells(row: dict, keys: list[str]) -> str:
    return ", ".join(
        f"{key}={cell(row[key])}" for key in keys if key in row and flat(row[key])
    )


def row_table(rows: list[dict], path: str, lines: list[str]) -> None:
    """A list of records, one line of scalars each and a walk into what nests."""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines.append(f"- `{path}` {len(rows)} rows")
    for index, row in enumerate(rows):
        cells = row_cells(row, keys)
        if cells:
            lines.append(f"    - [{index}] " + cells)
        # A row carrying its own dictionaries or record lists is walked, so no
        # scalar is lost to the row it happens to sit in.
        for key, value in row.items():
            if not flat(value):
                lines.extend(nested(value, f"{path}[{index}].{key}"))


def uniform_entities(node: dict) -> bool:
    """True where a dict is a table: several entities carrying the same fields."""
    values = list(node.values())
    if len(values) < MIN_UNIFORM or not all(isinstance(v, dict) for v in values):
        return False
    shared = set(values[0])
    if not shared:
        return False
    # Entities may carry an optional field or two; the fields they agree on have
    # to be most of what each of them holds.
    for entry in values[1:]:
        shared &= set(entry)
    return all(len(shared) >= 0.6 * len(entry) for entry in values)


def entity_table(node: dict, path: str, lines: list[str]) -> None:
    """A dict of like entities as one line each, in place of a flattened block."""
    entries = list(node.values())
    keys: list[str] = []
    for entry in entries:
        for key in entry:
            if key not in keys:
                keys.append(key)
    lines.append(
        f"- `{path}` {len(node)} entries, fields: " + ", ".join(f"`{k}`" for k in keys)
    )
    for name, entry in node.items():
        cells = row_cells(entry, keys)
        if cells:
            lines.append(f"    - **{name}**: " + cells)
        for key, value in entry.items():
            if not flat(value):
                lines.extend(nested(value, f"{path}.{name}.{key}"))


def walk(node, path: str, lines: list[str]) -> None:
    """Flatten one record: scalars verbatim, arrays as summaries."""
    if isinstance(node, dict):
        if path and uniform_entities(node):
            entity_table(node, path, lines)
            return
        for key, value in node.items():
            walk(value, f"{path}.{key}" if path else key, lines)
        return
    if isinstance(node, list):
        if not node:
            lines.append(f"- `{path}` empty")
            return
        if all(isinstance(item, dict) for item in node):
            if len(node) <= INLINE_ROWS:
                row_table(node, path, lines)
            else:
                lines.extend(column_census(node, path))
            return
        if len(node) <= INLINE_ARRAY and is_numeric(node):
            lines.append(f"- `{path}` " + ", ".join(number(v) for v in node))
            return
        if any(isinstance(item, (list, dict)) for item in node):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]", lines)
            return
        lines.append(array_line(path, node))
        return
    lines.append(
        f"- `{path}` = "
        + (number(node) if isinstance(node, (int, float)) else text(node))
    )


def stamp(record: dict) -> list[str]:
    """The geometry version and input digests, lifted out of the flattening."""
    out = []
    geometry = record.get("geometry")
    if isinstance(geometry, dict):
        parts = " ".join(f"{k}={v}" for k, v in geometry.items() if k != "geometry")
        out.append(f"geometry `{geometry.get('geometry', '?')}` [{parts}]")
    reads = record.get("reads")
    if isinstance(reads, dict):
        out.append("reads " + ", ".join(f"`{k}` {v}" for k, v in reads.items()))
    return out


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/DIGEST.md")
    parser.add_argument(
        "--root", action="append", default=None,
        help="directory of records to digest; repeatable",
    )
    arguments = parser.parse_args()

    roots = [Path(r) for r in (arguments.root or ROOTS)]
    files = sorted(
        path for root in roots for path in root.rglob("*.json")
    )

    lines: list[str] = [
        "# The records, digested",
        "",
        "Generated by `python tools/digest_results.py` over "
        + " and ".join(f"`{root}/`" for root in roots)
        + ". Every scalar of every record stands here verbatim; every array "
        "stands as its length, range, median and end points. Nothing is fitted "
        "or resampled, so an entry is either exact or explicitly a summary, and "
        "the record each came from is named above it for the cases where the "
        "elements themselves are wanted. Home directories in string values are "
        "written as `~`.",
        "",
    ]

    total = sum(p.stat().st_size for p in files)
    bundles = [p for p in files if p.as_posix() in BUNDLES]
    records = [p for p in files if p.as_posix() not in BUNDLES]
    bundle_bytes = sum(p.stat().st_size for p in bundles)
    lines += [
        f"{len(files)} files, {total / 1e6:.1f} MB. "
        f"{len(bundles)} export bundles carry {bundle_bytes / 1e6:.1f} MB of that "
        f"and are listed but never opened; the {len(records)} records below carry "
        f"{(total - bundle_bytes) / 1e3:.0f} kB.",
        "",
        "## Export bundles",
        "",
    ]
    for path in bundles:
        lines.append(
            f"- `{path.as_posix()}` {path.stat().st_size / 1e6:.1f} MB: "
            f"{BUNDLES[path.as_posix()]}"
        )
    for name, description in BUNDLES.items():
        if not Path(name).exists():
            lines.append(f"- `{name}` absent: {description}")
    lines += ["", "## Records", ""]

    for path in records:
        try:
            record = json.loads(path.read_text())
        except ValueError as error:
            lines += [f"### `{path.as_posix()}`", "", f"unreadable: {error}", ""]
            continue
        lines += [
            f"### `{path.as_posix()}`",
            "",
            f"{path.stat().st_size / 1e3:.1f} kB, sha256 `{digest_of(path)}`",
            "",
        ]
        for entry in stamp(record):
            lines.append(entry)
        if stamp(record):
            lines.append("")
        body = {k: v for k, v in record.items() if k not in ("geometry", "reads")}
        flattened: list[str] = []
        walk(body, "", flattened)
        lines += flattened
        lines.append("")

    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    size = destination.stat().st_size
    print(
        f"wrote {destination}, {size / 1e3:.0f} kB from "
        f"{(total - bundle_bytes) / 1e3:.0f} kB of records "
        f"({(total - bundle_bytes) / max(size, 1):.1f}x)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
