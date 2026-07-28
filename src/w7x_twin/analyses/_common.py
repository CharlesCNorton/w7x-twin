"""Shared infrastructure of the analysis entry points: session-cached inputs, positional
arguments, fixed-width tables, and geometry-stamped result records."""

from __future__ import annotations

import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from w7x_twin.hardware import walls
from w7x_twin.mhd.equilibrium import Twin
from w7x_twin.plasma import neoclassical

VESSEL_PART = "data/vessel.part"
COMPONENT_DIR = "data/pfc"


# -- arguments ---------------------------------------------------------------------

def arg(index: int, cast=str, default=None):
    """Positional argument ``index`` of the command, cast, or ``default`` when absent."""
    if len(sys.argv) > index:
        return cast(sys.argv[index])
    return default


def args(cast=str, start: int = 1) -> list:
    """Every positional argument from ``start`` on, cast; empty when none were given."""
    return [cast(value) for value in sys.argv[start:]]


# -- session-cached inputs ---------------------------------------------------------

@functools.lru_cache(maxsize=None)
def twin(coils_file: str = "coils.w7x") -> Twin:
    """The forward model, built once per process and shared between entry points."""
    return Twin(coils_file=coils_file, verbose=False)


@functools.lru_cache(maxsize=None)
def vessel() -> walls.Vessel:
    return walls.load_vessel(VESSEL_PART)


@functools.lru_cache(maxsize=None)
def components() -> list[walls.Component]:
    return walls.load_components(COMPONENT_DIR)


@functools.lru_cache(maxsize=None)
def drift_kinetic_coefficients():
    """The per-surface drift-kinetic tables if the radial scan exists, else the one
    solved surface."""
    profile = neoclassical.discover_monoenergetic_profile(
        Path(neoclassical.RADIAL_SCANS[0])
    )
    if profile is not None:
        return profile
    return neoclassical.load_monoenergetic(
        Path("cache/monkes_er.dat"), neoclassical.SINGLE_SURFACE
    )


@functools.lru_cache(maxsize=None)
def radial_coefficients():
    """The radial drift-kinetic scan, from the first directory carrying one."""
    for directory in neoclassical.RADIAL_SCANS:
        profile = neoclassical.discover_monoenergetic_profile(Path(directory))
        if profile is not None:
            return profile
    raise SystemExit("no drift-kinetic scan found")


@functools.lru_cache(maxsize=None)
def ripple() -> neoclassical.EffectiveRipple:
    return neoclassical.load_ripple()


def layer_constants() -> tuple[float, float, float]:
    """(connection length, incidence sine, wetted area) from the exhaust record."""
    record = Path("results/exhaust/heat_flux.json")
    if record.exists():
        stored = json.loads(record.read_text())
        return (
            float(stored.get("connection_length_m", 180.0)),
            float(stored.get("incidence_sine", 0.035)),
            float(stored.get("wetted", {}).get("area_m2", 0.72)),
        )
    return 180.0, 0.035, 0.72


# -- output ------------------------------------------------------------------------

def _encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def file_digest(path: str | Path) -> str:
    """Content digest of one file, or the marker for a file that is not there."""
    path = Path(path)
    if not path.is_file():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_record(
    path: str | Path, payload: dict, geometry=None, note: str = "",
    reads: "tuple[str | Path, ...] | None" = None,
) -> Path:
    """Write one results record, stamped with the geometry it was computed under and
    with the digests of the records it read, so a stale dependency is detectable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict = {"geometry": geometry.as_dict()} if geometry is not None else {}
    if reads is not None:
        body["reads"] = {str(source): file_digest(source) for source in reads}
    body.update(payload)
    path.write_text(json.dumps(body, indent=2, default=_encode))
    print(f"\nwrote {path}{note}")
    return path


class Table:
    """A fixed-width table whose layout is stated once: ``(title, value_format)`` per
    column, string formats left-aligned and numeric ones right-aligned."""

    def __init__(self, *columns: tuple[str, str]) -> None:
        self.formats = [spec for _, spec in columns]
        cells = []
        for title, spec in columns:
            width = self._width(spec)
            align = "<" if spec.endswith("s") and not spec.startswith(">") else ">"
            cells.append(f"{title:{align}{width}s}" if width else title)
        self.header = " ".join(cells)

    @staticmethod
    def _width(spec: str) -> int:
        digits = ""
        for character in spec.lstrip("<>"):
            if not character.isdigit():
                break
            digits += character
        return int(digits or 0)

    def begin(self, extra: int = 0) -> None:
        print(self.header)
        print("-" * (len(self.header) + extra))

    def row(self, *values) -> None:
        print(
            " ".join(
                f"{value:{spec}}"
                for value, spec in zip(values, self.formats, strict=True)
            ),
            flush=True,
        )
