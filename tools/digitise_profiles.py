"""Digitise the Thomson and charge-exchange profile figures of the pellet paper.

The figures are vector graphics, so the plotted curve is a polyline and the shaded band
around it is a filled polygon, both with their coordinates in the file. The tick labels
carry their own positions, so the pixel-to-data map is read off the figure and its residual
against a straight line through those ticks is what the digitisation is accurate to.
"""

import json
import sys
from pathlib import Path

import fitz
import numpy as np


def paths(page, box, min_points=40):
    """Long paths whose points fall inside a box, as (points, colour, filled)."""
    out = []
    for group in page.get_drawings():
        points = []
        for item in group["items"]:
            if item[0] == "l":
                points.extend([(item[1].x, item[1].y), (item[2].x, item[2].y)])
            elif item[0] == "c":
                points.extend([(item[k].x, item[k].y) for k in (1, 2, 3, 4)])
        if len(points) < min_points:
            continue
        a = np.array(points)
        inside = (
            (a[:, 0] >= box[0] - 2) & (a[:, 0] <= box[2] + 2)
            & (a[:, 1] >= box[1] - 2) & (a[:, 1] <= box[3] + 2)
        )
        if inside.mean() < 0.9:
            continue
        colour = group.get("fill") or group.get("color")
        out.append((a[inside], tuple(round(v, 3) for v in colour), group.get("fill") is not None))
    return out


def axis_map(ticks):
    """Pixel-to-data map through labelled ticks, and how far from linear they are."""
    pixels = np.array([p for p, _ in ticks], dtype=float)
    values = np.array([v for _, v in ticks], dtype=float)
    order = np.argsort(pixels)
    pixels, values = pixels[order], values[order]
    slope, intercept = np.polyfit(pixels, values, 1)
    residual = float(np.max(np.abs(values - (slope * pixels + intercept))))

    def to_data(x):
        return np.interp(np.asarray(x, dtype=float), pixels, values)

    return to_data, residual, float(abs(slope))


def labelled(page, want_y=None, want_x=None, tolerance=3.0, within=None):
    """Numeric labels sharing a row or a column, as (pixel centre, value)."""
    out = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        try:
            value = float(text)
        except ValueError:
            continue
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if want_y is not None and abs(cy - want_y) <= tolerance:
            if within is None or within[0] <= cx <= within[1]:
                out.append((cx, value))
        if want_x is not None and abs(cx - want_x) <= tolerance:
            if within is None or within[0] <= cy <= within[1]:
                out.append((cy, value))
    return out


def resample(points, x_of, y_of, samples=41):
    """A path as (x, y) in data coordinates on a uniform grid in x."""
    order = np.argsort(points[:, 0])
    px, py = points[order, 0], points[order, 1]
    x = x_of(px)
    y = y_of(py)
    grid = np.linspace(float(x.min()), float(x.max()), samples)
    return grid, np.interp(grid, x, y)


def band(points, x_of, y_of, samples=41):
    """Lower and upper envelope of a filled band, on a uniform grid in x."""
    x = x_of(points[:, 0])
    y = y_of(points[:, 1])
    grid = np.linspace(float(x.min()), float(x.max()), samples)
    low, high = [], []
    half = 0.5 * (grid[1] - grid[0]) if len(grid) > 1 else 1.0
    for centre in grid:
        near = np.abs(x - centre) <= max(half, 1e-9)
        if not near.any():
            near = np.abs(x - centre) <= 3.0 * half
        low.append(float(np.min(y[near])) if near.any() else float("nan"))
        high.append(float(np.max(y[near])) if near.any() else float("nan"))
    return grid, np.array(low), np.array(high)


SOURCE = (
    "S. A. Bozhenkov et al., High-performance plasmas after pellet injections in "
    "Wendelstein 7-X, Nucl. Fusion 60, 066011 (2020), open copy at pure.mpg.de "
    "item_3231111, digitised from the vector paths of the published figures"
)


def main() -> int:
    document = fitz.open(sys.argv[1])
    figures = []

    # Figure 6: electron density against the normalised radius on the upper axis, which is
    # two-sided about the magnetic axis and whose inboard labels are printed unsigned.
    page = document[11]
    ticks = labelled(page, want_y=109.1, tolerance=3.0, within=(200.0, 400.0))
    zero = min(ticks, key=lambda t: t[1])[0]
    ticks = [(p, v if p >= zero else -v) for p, v in ticks]
    rho_of, rho_residual, rho_scale = axis_map(ticks)
    density_of, density_residual, _ = axis_map(
        labelled(page, want_x=188.5, tolerance=3.0, within=(130.0, 280.0))
    )
    box = (205.0, 100.0, 408.0, 276.0)
    series = []
    for points, colour, filled in paths(page, box):
        if filled:
            continue
        red = colour[0] > 0.5
        x, y = resample(points, rho_of, density_of)
        series.append(
            {
                "label": "post-pellet" if red else "pre-pellet, doubled in the figure",
                "phase_s": [1.65, 1.75] if red else [0.35, 0.42],
                # The figure states the pre-pellet profile is drawn at twice its value.
                "scale_applied_in_figure": 1.0 if red else 2.0,
                "x": [float(v) for v in x],
                "y": [float(v) for v in (y if red else y / 2.0)],
            }
        )
    figures.append(
        {
            "figure": 6,
            "discharge": "20181016.037",
            "quantity": "electron density",
            "unit": "1e19 m^-3",
            "abscissa": "rho = r_eff / r_lcfs, signed inboard to outboard",
            "axis_residual": {"abscissa": rho_residual, "ordinate": density_residual},
            "series": series,
        }
    )

    # Figure 7: electron and ion temperature against the effective radius, two panels, each
    # with a shaded band the paper draws as the profile's own uncertainty.
    page = document[12]
    for panel, box, discharge, phase in (
            ("a", (110.0, 96.0, 292.0, 218.0), "20181016.037", [1.67, 1.75]),
            ("b", (319.0, 96.0, 500.0, 218.0), "20180920.017", None),
    ):
        radius_of, radius_residual, _ = axis_map(
            labelled(page, want_y=225.0, tolerance=6.0, within=(box[0] - 8, box[2] + 8))
        )
        temperature_of, temperature_residual, _ = axis_map(
            labelled(page, want_x=box[0] + 4.0, tolerance=6.0, within=(box[1], box[3] + 4))
        )
        curves, bands = [], []
        for points, colour, filled in paths(page, box):
            (bands if filled else curves).append((points, colour))
        series = []
        for points, colour in curves:
            electron = max(colour) < 0.35
            x, y = resample(points, radius_of, temperature_of)
            entry = {
                "label": "electron, Thomson scattering" if electron
                else "ion, charge exchange recombination spectroscopy",
                "x": [float(v) for v in x],
                "y": [float(v) for v in y],
            }
            # The band drawn in the same colour family is that profile's uncertainty.
            for band_points, band_colour in bands:
                pale = min(band_colour) > 0.6
                is_grey = max(band_colour) - min(band_colour) < 0.05
                if pale and (is_grey == electron):
                    bx, low, high = band(band_points, radius_of, temperature_of)
                    entry["band_x"] = [float(v) for v in bx]
                    entry["band_low"] = [float(v) for v in low]
                    entry["band_high"] = [float(v) for v in high]
                    break
            series.append(entry)
        figures.append(
            {
                "figure": 7,
                "panel": panel,
                "discharge": discharge,
                "phase_s": phase,
                "quantity": "temperature",
                "unit": "keV",
                "abscissa": "r_eff in metres",
                "axis_residual": {
                    "abscissa": radius_residual, "ordinate": temperature_residual
                },
                "series": series,
            }
        )

    Path(sys.argv[2]).write_text(
        json.dumps({"source": SOURCE, "figures": figures}, indent=1)
    )
    for figure in figures:
        print(
            f"figure {figure['figure']}{figure.get('panel', '')} {figure['discharge']}: "
            f"axis residual {figure['axis_residual']['abscissa']:.2e} / "
            f"{figure['axis_residual']['ordinate']:.2e}"
        )
        for entry in figure["series"]:
            y = entry["y"]
            note = " with a band" if "band_low" in entry else ""
            print(
                f"  {entry['label']:52s} {len(y)} points, "
                f"{min(y):.3f} to {max(y):.3f}{note}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
