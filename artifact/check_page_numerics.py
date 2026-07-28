"""Run the page's own tracer against the model's answer for the same grid.

The rendered page carries its own field contraction, its own Runge-Kutta step, its own
fixed-point search for the magnetic axis and its own flux-surface reconstruction, written in
JavaScript. None of that is exercised by the Python tests. This extracts those functions
from the template, runs them under node against the exported field bundle, and compares the
result with what `python -m w7x_twin page-error` measured on the same coarsened grid in
Python, so the two implementations are checked against each other rather than assumed to
agree.

    check_page_numerics.py [template] [field bundle] [reference json]

Exits non-zero when they disagree by more than the tolerance below.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

#: Agreement required between the two implementations. Both integrate the same equations on
#: the same grid, so they should differ only by float32 storage and accumulation order.
AXIS_TOLERANCE_M = 1.0e-4
IOTA_TOLERANCE = 1.0e-4

#: The blocks the harness needs, in the order the template defines them, taken verbatim
#: from it. The order matters: the resolution constants read the field instance.
BLOCKS = (
    ("array types", r"const ARRAY_TYPES = \{.*?\n\};"),
    ("decode", r"function decode\(entry\) \{.*?\n\}"),
    ("vacuum field", r"class VacuumField \{.*?\n\}\n"),
    (
        "field instance",
        r"const field = new VacuumField\(FIELD\);\nconst derivative = "
        r"new Float32Array\(3\);",
    ),
    ("step", r"function step\(state, phi, dphi\) \{.*?\n\}"),
    ("resolution", r"const STEPS_PER_PERIOD = .*?const DPHI = [^;]*;"),
    ("find axis", r"function findAxis\(guessR, guessZ\) \{.*?\n\}"),
)

HARNESS = """
import { readFileSync } from "node:fs";
const FIELD = JSON.parse(readFileSync(process.argv[2], "utf8"));
%(blocks)s

const currents = FIELD.presets[0].currents.slice();
field.setCurrents(currents, 0, field.aux ? field.aux.circuits.map(() => 0) : null);

const axis = findAxis(5.93, 0.0);
if (!axis) { console.log(JSON.stringify({ error: "axis not found" })); process.exit(0); }

/* The winding number of one line about the co-traced axis, the way the page measures the
   transform. */
function winding(startR, turns) {
  const s = [startR, axis[1]];
  const a = [axis[0], axis[1]];
  let total = 0;
  let previous = Math.atan2(s[1] - a[1], s[0] - a[0]);
  for (let n = 0; n < turns * STEPS_PER_TURN; n++) {
    const phi = n * DPHI;
    if (!step(s, phi, DPHI)) return NaN;
    if (!step(a, phi, DPHI)) return NaN;
    const angle = Math.atan2(s[1] - a[1], s[0] - a[0]);
    let delta = angle - previous;
    if (delta > Math.PI) delta -= 2 * Math.PI;
    if (delta < -Math.PI) delta += 2 * Math.PI;
    total += delta;
    previous = angle;
  }
  return Math.abs(total) / (2 * Math.PI * turns);
}

const halfWidth = Number(process.argv[3]);
const layer = [Number(process.argv[4]), Number(process.argv[5])];
const lines = Number(process.argv[6]);
const turns = Number(process.argv[7]);
const iota = [];
for (let k = 0; k < lines; k++) {
  const t = k / (lines - 1);
  iota.push(winding(axis[0] + (layer[0] + (layer[1] - layer[0]) * t) * halfWidth, turns));
}

/* The plasma-current field must move the axis inward when it is switched on: it is
   diamagnetic, and its Shafranov shift is outward in major radius. */
let shifted = null;
if (field.plasma) {
  field.setCurrents(currents, field.plasma.betaReference,
    field.aux ? field.aux.circuits.map(() => 0) : null);
  shifted = findAxis(axis[0], axis[1]);
  field.setCurrents(currents, 0, field.aux ? field.aux.circuits.map(() => 0) : null);
}

/* One energised trim coil must break the five-fold periodicity of |B| on a midplane
   circle, and must not when it is off. */
function harmonics(auxCurrents) {
  field.setCurrents(currents, 0, auxCurrents);
  const points = 90;
  const values = [];
  for (let i = 0; i < points; i++) {
    const phi = (2 * Math.PI * i) / points;
    if (!field.at(6.2, phi, 0.0, derivative)) return null;
    values.push(derivative[0]);
  }
  const out = [];
  for (let n = 0; n <= 6; n++) {
    let re = 0, im = 0;
    for (let i = 0; i < points; i++) {
      const angle = (2 * Math.PI * n * i) / points;
      re += values[i] * Math.cos(angle);
      im -= values[i] * Math.sin(angle);
    }
    out.push((2 * Math.hypot(re, im)) / points);
  }
  return out;
}

let trimOff = null, trimOn = null;
if (field.aux) {
  const zero = field.aux.circuits.map(() => 0);
  trimOff = harmonics(zero);
  const one = field.aux.circuits.map((c) => (c.name === "trim_a1" ? 1800 : 0));
  trimOn = harmonics(one);
  field.setCurrents(currents, 0, zero);
}

console.log(JSON.stringify({
  axis_r: axis[0], axis_z: axis[1],
  iota: iota.map((v) => (Number.isFinite(v) ? v : null)),
  axis_with_plasma: shifted ? shifted[0] : null,
  trim_off: trimOff, trim_on: trimOn,
}));
"""


def extract(template: str) -> str:
    parts = []
    for label, pattern in BLOCKS:
        match = re.search(pattern, template, re.S)
        if match is None:
            raise SystemExit(f"the template no longer carries the {label} block")
        parts.append(match.group(0))
    return "\n".join(parts)


def main() -> int:
    template_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "twin3d.template.html"
    field_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/magnetics/w7x_field.json")
    reference_path = (
        Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/archive/page_tracer_error.json")
    )
    for path in (template_path, field_path, reference_path):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    template = template_path.read_text(encoding="utf-8")
    script = HARNESS % {"blocks": extract(template)}
    reference = json.loads(reference_path.read_text())
    page = next(row for row in reference["variants"] if row["variant"] == "page")

    # The same fan the Python measurement launched: the same fractions of the same
    # axis-to-boundary span, from each implementation's own axis.
    layer = reference["page_layer"]
    turns = reference["page_turns"]
    lines = reference["page_lines"]
    half_width = reference["half_width_m"]

    with tempfile.TemporaryDirectory() as work:
        harness = Path(work) / "check.mjs"
        harness.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [
                "node", str(harness), str(field_path), f"{half_width}",
                f"{layer[0]}", f"{layer[1]}", f"{lines}", f"{turns}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(f"node failed:\n{completed.stdout}\n{completed.stderr}")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if "error" in result:
        raise SystemExit(f"the page's tracer reports: {result['error']}")

    checks: list[tuple[str, bool, str]] = []

    axis_gap = abs(result["axis_r"] - page["axis_r"])
    checks.append(
        (
            "magnetic axis against the Python tracer on the same grid",
            axis_gap <= AXIS_TOLERANCE_M,
            f"{result['axis_r']:.6f} m against {page['axis_r']:.6f} m, "
            f"{1e6 * axis_gap:.1f} um apart",
        )
    )
    # Per line over the contiguous block both implementations trace from the axis out.
    # The fan is laid on the axis-to-LCFS span, so every line inside the span is regular
    # and the two tracers must agree there; past the LCFS a line may leave the grid, and
    # where one implementation loses it the comparison stops rather than fails.
    measured = [float("nan") if v is None else float(v) for v in result["iota"]]
    stored = [float(v) for v in page["iota_lines"]]
    block = 0
    for ours, theirs in zip(measured, stored):
        if not (math.isfinite(ours) and math.isfinite(theirs)):
            break
        block += 1
    inside = sum(
        1 for k in range(lines)
        if layer[0] + (layer[1] - layer[0]) * k / (lines - 1) <= 1.0
    )
    checks.append(
        (
            "the fan traced to the LCFS in both implementations",
            block >= inside,
            f"{block} of {lines} lines in the common block, {inside} inside the span",
        )
    )
    compared = min(block, inside)
    iota_gap = max(
        (abs(a - b) for a, b in zip(measured[:compared], stored[:compared])),
        default=float("inf"),
    )
    checks.append(
        (
            "winding numbers across the fan inside the LCFS",
            iota_gap <= IOTA_TOLERANCE,
            f"largest gap {iota_gap:.2e} over {compared} lines",
        )
    )

    if result["axis_with_plasma"] is not None:
        shift = 1e3 * (result["axis_with_plasma"] - result["axis_r"])
        checks.append(
            (
                "the plasma current shifts the axis outward",
                shift > 0.0,
                f"{shift:+.2f} mm at the reference pressure",
            )
        )

    if result["trim_off"] is not None:
        off = result["trim_off"]
        on = result["trim_on"]
        periodic = max(off[1:5]) / max(off[5], 1e-30)
        checks.append(
            (
                "the trim circuits leave n = 1 to 4 empty when unpowered",
                periodic < 1e-6,
                f"n = 1 to 4 at {max(off[1:5]):.2e} T against n = 5 at {off[5]:.2e} T",
            )
        )
        checks.append(
            (
                "one energised trim coil populates n = 1",
                on[1] > 1e-4,
                f"{1e3 * on[1]:.3f} mT at 1800 A per turn",
            )
        )

    print(f"the page's numerics against the model, field bundle {field_path}")
    for name, passed, detail in checks:
        print(f"  {'ok  ' if passed else '??  '}{name:58s} {detail}")
    failed = [name for name, passed, _ in checks if not passed]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} agree")
    for name in failed:
        print(f"  disagrees: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
