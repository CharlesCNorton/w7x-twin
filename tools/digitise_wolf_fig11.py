"""Digitise the raster figure 11 of Wolf 2019 into time-resolved 20180919.033 profiles.

    python tools/digitise_wolf_fig11.py <wolf2019.pdf> <output.json>
"""

import json
import sys
from pathlib import Path

import fitz
import numpy as np

#: Page carrying the figure, and the series times and colours drawn in its legends.
PAGE_INDEX = 14
TIMES_S = (1.80, 2.80, 3.50, 4.30)

#: Panel definitions: (quantity, unit, x ticks (min, max, count), y ticks, abscissa).
PANELS = (
    ("electron density", "1e19 m^-3", (0.0, 1.5, 4), (0.0, 20.0, 5), "position along laser [m]"),
    ("electron temperature", "keV", (0.0, 1.5, 4), (0.0, 1.5, 4), "position along laser [m]"),
    ("ion temperature", "keV", (5.4, 5.9, 6), (0.0, 2.0, 5), "R in NBI plane [m]"),
)


def axis_map(ticks_px: np.ndarray, frame: tuple[float, float], spec, descending: bool):
    """Pixel-to-value map with the tick lattice offset fixed by frame containment:
    the mapped frame must cover the full tick range, which selects the offset uniquely."""
    low, high, count = spec
    step = (high - low) / (count - 1)
    spacing = float(np.median(np.diff(ticks_px))) if ticks_px.size > 1 else None
    if spacing is None or spacing <= 0:
        raise SystemExit("axis ticks too sparse to fit")
    base = np.round((ticks_px - ticks_px[0]) / spacing)
    for offset in range(0, count):
        indices = base + offset
        if indices.max() > count - 1:
            break
        values = (high - indices * step) if descending else (low + indices * step)
        slope, intercept = np.polyfit(ticks_px, values, 1)
        ends = sorted(slope * np.array(frame) + intercept)
        slack = 0.1 * step
        if ends[0] <= low + slack and ends[1] >= high - slack:
            return lambda px: slope * np.asarray(px, dtype=float) + intercept
    raise SystemExit("no tick-lattice offset satisfies frame containment")


def figure_pixels(pdf: Path) -> np.ndarray:
    """The figure's raster as an (h, w, 3) uint8 array."""
    document = fitz.open(pdf)
    page = document[PAGE_INDEX]
    xref = page.get_images(full=True)[0][0]
    pix = fitz.Pixmap(document, xref)
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    return data.reshape(pix.height, pix.width, pix.n)[:, :, :3]


def frame_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Plot frames as (x0, y0, x1, y1), from the long dark vertical rules."""
    dark = (image < 90).all(axis=2)
    column_run = dark.sum(axis=0)
    tall = np.where(column_run > 0.6 * image.shape[0])[0]
    # Cluster adjacent columns into rules, then pair them into frames.
    rules = []
    for column in tall:
        if rules and column - rules[-1][-1] <= 3:
            rules[-1].append(column)
        else:
            rules.append([column])
    edges = [int(np.mean(rule)) for rule in rules]
    boxes = []
    for left, right in zip(edges[0::2], edges[1::2]):
        rows = np.where(dark[:, left + 3 : right - 3].mean(axis=1) > 0.6)[0]
        boxes.append((left, int(rows.min()), right, int(rows.max())))
    return boxes


def series_colours(image: np.ndarray, box) -> list[tuple[int, int, int]]:
    """The four legend-dot colours in legend order, top dot first."""
    x0, y0, x1, y1 = box
    strip = image[y0 + 2 : y1 - 2, x0 + int(0.72 * (x1 - x0)) : x1 - 2].astype(int)
    chroma = strip.max(axis=2) - strip.min(axis=2)
    mask = (chroma > 60) & (strip.sum(axis=2) < 620)
    rows = np.where(mask.any(axis=1))[0]
    groups = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)
    dots = [group for group in groups if len(group) >= 6][:4]
    colours = []
    for group in dots:
        pixels = strip[group][mask[group]]
        colours.append(tuple(int(v) for v in np.median(pixels, axis=0)))
    return colours


def tick_positions(image: np.ndarray, box) -> tuple[np.ndarray, np.ndarray]:
    """Pixel positions of the major x and y ticks: the deep inward marks, else the shallow ones."""
    x0, y0, x1, y1 = box
    dark = (image < 90).all(axis=2)

    def scan(strip, offset):
        hits = np.where(strip.mean(axis=1) > 0.3)[0]
        groups = np.split(hits, np.where(np.diff(hits) > 3)[0] + 1)
        return np.array([float(np.mean(g)) + offset for g in groups if g.size])

    y_ticks = scan(dark[y0 + 4 : y1 - 3, x0 + 8 : x0 + 13], y0 + 4)
    if y_ticks.size < 2:
        y_ticks = scan(dark[y0 + 4 : y1 - 3, x0 + 2 : x0 + 7], y0 + 4)
    x_ticks = scan(dark[y1 - 12 : y1 - 7, x0 + 4 : x1 - 3].T, x0 + 4)
    if x_ticks.size < 2:
        x_ticks = scan(dark[y1 - 6 : y1 - 1, x0 + 4 : x1 - 3].T, x0 + 4)
    return x_ticks, y_ticks


def curve_of_colour(image, box, colour, tolerance=60):
    """The thin fitted curve of one colour per column, tracked by linear prediction."""
    x0, y0, x1, y1 = box
    region = image[y0:y1, x0:x1].astype(int)
    distance = np.abs(region - np.array(colour)).sum(axis=2)
    mask = distance < tolerance
    xs, ys = [], []
    missed = 0
    for column in range(mask.shape[1]):
        rows = np.where(mask[:, column])[0]
        if rows.size == 0:
            missed += 1
            continue
        splits = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
        thin = [run for run in splits if len(run) <= 9]
        if not thin:
            missed += 1
            continue
        centres = [float(np.mean(run)) for run in thin]
        if len(xs) >= 2 and missed <= 15:
            slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1.0)
            predicted = ys[-1] + slope * (column - xs[-1])
            pick = min(centres, key=lambda c: abs(c - predicted))
            if abs(pick - predicted) > 40:
                missed += 1
                continue
        elif len(xs) >= 1 and missed <= 15:
            pick = min(centres, key=lambda c: abs(c - ys[-1]))
            if abs(pick - ys[-1]) > 40:
                missed += 1
                continue
        else:
            pick = float(np.median(centres))
        missed = 0
        xs.append(float(column))
        ys.append(pick)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def shading_window(image, box) -> tuple[float, float]:
    """Pixel columns where the grey r_eff shading ends and resumes."""
    x0, y0, x1, y1 = box
    band = image[(y0 + y1) // 2 - 4 : (y0 + y1) // 2 + 4, x0:x1].astype(int)
    grey = (
        (np.abs(band[:, :, 0] - band[:, :, 1]) < 8)
        & (np.abs(band[:, :, 1] - band[:, :, 2]) < 8)
        & (band[:, :, 0] > 180)
        & (band[:, :, 0] < 235)
    ).mean(axis=0) > 0.5
    inside = np.where(~grey)[0]
    return float(inside.min()), float(inside.max())


def main() -> int:
    pdf, output = Path(sys.argv[1]), Path(sys.argv[2])
    image = figure_pixels(pdf)
    boxes = frame_boxes(image)
    if len(boxes) != 3:
        raise SystemExit(f"expected three panel frames, found {len(boxes)}")

    figures = []
    for box, (quantity, unit, x_range, y_range, abscissa) in zip(boxes, PANELS):
        x0, y0, x1, y1 = box
        colours = series_colours(image, box)
        x_ticks, y_ticks = tick_positions(image, box)
        if x_ticks.size < 2 or y_ticks.size < 2:
            raise SystemExit(f"{quantity}: ticks not found")
        x_of = axis_map(x_ticks, (float(x0), float(x1)), x_range, descending=False)
        y_of = axis_map(y_ticks, (float(y0), float(y1)), y_range, descending=True)

        def to_data(xs, ys):
            return x_of(np.asarray(xs) + x0), y_of(np.asarray(ys) + y0)

        window = shading_window(image, box)
        window_x = tuple(float(x_of(w + x0)) for w in window)
        series = []
        for colour, time in zip(colours, TIMES_S):
            xs, ys = curve_of_colour(image, box, colour)
            if xs.size < 40:
                raise SystemExit(
                    f"{quantity} at {time} s traced only {xs.size} columns"
                )
            x, y = to_data(xs, ys)
            series.append(
                {
                    "label": f"{time:.2f} s",
                    "time_s": time,
                    "x": [round(float(v), 5) for v in x],
                    "y": [round(float(v), 5) for v in y],
                }
            )
        figures.append(
            {
                "figure": "11",
                "discharge": "20180919.033",
                "quantity": quantity,
                "unit": unit,
                "abscissa": abscissa,
                "core_window": [round(v, 4) for v in window_x],
                "series": series,
            }
        )

    output.write_text(json.dumps({"figures": figures}, indent=1))
    print(f"wrote {output} with {sum(len(f['series']) for f in figures)} curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
