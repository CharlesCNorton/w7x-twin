"""Re-verify the saturation response against the growth-rate grid, and re-stamp it.

The response record stores, per nonlinear run, the linear drive that run answered
to: ``mixing_length_sum`` over the whole spectrum and ``drive_in_box`` over the
band its own perpendicular box carried. Both are read out of the growth-rate grid
rather than measured, so extending the grid with rows the response never read
changes the grid's digest without changing a single number in the record.

This recomputes both from the grid as it stands and compares them against what the
record holds. Every point matching exactly means the response still answers to the
grid it names, and the recorded input digest is refreshed to say so. Any point
moving means the response was measured against a drive that no longer exists, and
the record is left alone for `w7x-twin saturation` to remeasure.

    python tools/reverify_saturation.py [--write]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w7x_twin.analyses import _common
from w7x_twin.analyses.turbulence import (
    CONSTANT_RECORD,
    DENSITY_GRADIENT,
    GRID_RECORD,
    NX,
    NY,
    Y0,
    box_wavenumbers,
)
from w7x_twin.plasma.transport import GrowthRateTable


def derived(table: GrowthRateTable, point: dict) -> dict[str, float]:
    """The two quantities a stored point takes from the grid."""
    surface = float(point["torflux"])
    gradient = float(point["gradient"])
    density = float(point.get("density_gradient", DENSITY_GRADIENT))
    total = table.mixing_length_sum(surface, gradient, density)

    low, high = box_wavenumbers(
        float(point.get("y0", Y0)), int(point.get("box", [NX, NY])[1])
    )
    in_box = table.mixing_length_sum(
        surface, gradient, density, ky_max=high
    ) - table.mixing_length_sum(
        surface, gradient, density, ky_max=low * (1.0 - 1e-9)
    )
    return {"mixing_length_sum": total, "drive_in_box": in_box}


def main() -> int:
    write = "--write" in sys.argv
    for path in (GRID_RECORD, CONSTANT_RECORD):
        if not path.is_file():
            raise SystemExit(f"no record at {path}")

    stored = json.loads(CONSTANT_RECORD.read_text())
    points = stored.get("points", [])
    if not points:
        raise SystemExit(f"{CONSTANT_RECORD} carries no points")

    configuration = points[0].get("configuration")
    table = GrowthRateTable.read(GRID_RECORD, configuration)

    layout = _common.Table(
        ("s", "6.2f"), ("a/L_T", "6.1f"), ("a/L_n", "6.1f"), ("box", ">9s"),
        ("quantity", "18s"), ("stored", "12.6f"), ("now", "12.6f"), ("verdict", ">8s"),
    )
    layout.begin()
    moved = []
    compared = 0
    for point in points:
        now = derived(table, point)
        for key, value in now.items():
            held = point.get(key)
            if held is None:
                continue
            compared += 1
            agrees = float(held) == float(value)
            if not agrees:
                moved.append((point, key, float(held), float(value)))
            layout.row(
                float(point["torflux"]), float(point["gradient"]),
                float(point.get("density_gradient", DENSITY_GRADIENT)),
                "x".join(str(v) for v in point.get("box", [NX, NY])),
                key, float(held), float(value), "same" if agrees else "MOVED",
            )

    print()
    print(
        f"{compared - len(moved)} of {compared} stored drives reproduce from the grid "
        f"as it stands"
    )
    if moved:
        print("the response answers to a drive the grid no longer carries:")
        for point, key, held, value in moved:
            print(
                f"  s = {point['torflux']}, a/L_T = {point['gradient']}: "
                f"{key} {held:.6g} against {value:.6g}"
            )
        print(f"re-measure with `w7x-twin saturation`; {CONSTANT_RECORD} left as it stands")
        return 1

    digest = _common.file_digest(GRID_RECORD)
    recorded = (stored.get("reads") or {}).get(GRID_RECORD.as_posix())
    if recorded == digest:
        print(f"the recorded input digest is already {digest}")
        return 0
    print(f"recorded input digest {recorded} against {digest}")
    if not write:
        print("pass --write to refresh it")
        return 0

    stored["reads"] = {GRID_RECORD.as_posix(): digest}
    CONSTANT_RECORD.write_text(json.dumps(stored, indent=2))
    print(f"refreshed the input digest of {CONSTANT_RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
