"""Coil circuits, current configurations, in-vessel epochs, geometry versioning, and accumulated load."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import numpy as np

# -- from machine ------------------------------------------------------------------

# Coil circuits and field grid from the IPP single-filament model coils.w7x (MAKEGRID format).

# -- machine constants --------------------------------------------------------------

NUM_FIELD_PERIODS = 5
MAJOR_RADIUS_M = 5.5
MINOR_RADIUS_M = 0.53
NOMINAL_FIELD_T = 2.5
NOMINAL_PLASMA_VOLUME_M3 = 30.0

#: Maximum current per turn of the superconducting modular coils (NbTi, 3.9 K).
MAX_COIL_CURRENT_A = 17600.0


@dataclasses.dataclass(frozen=True)
class Circuit:
    """One independently powered circuit; ``extcur`` is amperes per turn and ``turns * extcur`` the amp-turns."""

    key: str
    label: str
    turns: int
    num_coils: int
    kind: str


#: The seven main circuits of the superconducting magnet system: five non-planar
#: modular coil types and two planar coil types, ten coils each.
MAIN_CIRCUITS: tuple[Circuit, ...] = (
    Circuit("npc1", "Non-planar coil 1", 108, 10, "non-planar"),
    Circuit("npc2", "Non-planar coil 2", 108, 10, "non-planar"),
    Circuit("npc3", "Non-planar coil 3", 108, 10, "non-planar"),
    Circuit("npc4", "Non-planar coil 4", 108, 10, "non-planar"),
    Circuit("npc5", "Non-planar coil 5", 108, 10, "non-planar"),
    Circuit("pca", "Planar coil A", 36, 10, "planar"),
    Circuit("pcb", "Planar coil B", 36, 10, "planar"),
)

#: The ten in-vessel control coils and five trim coils, declared without geometry until a coils file supplies it.
AUXILIARY_CIRCUITS: tuple[Circuit, ...] = (
    # Ten independently powered coils, two per module. Driving the five upper together and
    # the five lower together can only carry a periodic pattern, so the 2/2 correction the
    # machine applies needs them separated.
    *(
        Circuit(f"cc{module}u", f"Control coil, module {module} upper", 8, 1, "control")
        for module in range(1, NUM_FIELD_PERIODS + 1)
    ),
    *(
        Circuit(f"cc{module}l", f"Control coil, module {module} lower", 8, 1, "control")
        for module in range(1, NUM_FIELD_PERIODS + 1)
    ),
    Circuit("trim_a1", "Trim coil A1", 48, 1, "trim"),
    Circuit("trim_a2", "Trim coil A2", 48, 1, "trim"),
    Circuit("trim_a3", "Trim coil A3", 48, 1, "trim"),
    Circuit("trim_a4", "Trim coil A4", 48, 1, "trim"),
    Circuit("trim_b1", "Trim coil B1 (type B)", 72, 1, "trim"),
)


#: Order the trim circuits are driven in as the n = 1 waveform steps round the machine,
#: module 1 first. The type B coil sits in module 5.
TRIM_ORDER: tuple[str, ...] = ("trim_a1", "trim_a2", "trim_a3", "trim_a4", "trim_b1")


def trim_waveform(amplitude_a: float, phase_degrees: float) -> dict[str, float]:
    """Trim currents I_k = A0 cos(2 pi (k - 1) / 5 - phi) in the published convention, in A per turn."""
    phase = np.radians(phase_degrees)
    return {
        key: float(amplitude_a * np.cos(2.0 * np.pi * index / NUM_FIELD_PERIODS - phase))
        for index, key in enumerate(TRIM_ORDER)
    }


@dataclasses.dataclass(frozen=True)
class FieldGrid:
    """Cylindrical grid the vacuum field response is tabulated on."""

    r_min: float
    r_max: float
    num_r: int
    z_min: float
    z_max: float
    num_z: int
    num_phi: int
    num_field_periods: int
    stellarator_symmetric: bool
    normalize_by_currents: bool

    @property
    def num_cells(self) -> int:
        return self.num_r * self.num_z * self.num_phi


@dataclasses.dataclass
class CoilSet:
    """Filament geometry of one coils file, grouped by circuit."""

    path: Path
    grid: FieldGrid
    circuit_keys: list[str]
    group_names: list[str]
    #: filaments[i] is a list of (n_points, 3) arrays of XYZ winding-pack points.
    filaments: list[list[np.ndarray]]
    #: Current value in the coils file for each circuit, i.e. the winding number.
    file_currents: list[float]

    @property
    def num_circuits(self) -> int:
        return len(self.circuit_keys)

    def circuits(self) -> list[Circuit]:
        by_key = {c.key: c for c in MAIN_CIRCUITS + AUXILIARY_CIRCUITS}
        return [by_key[k] for k in self.circuit_keys]

    def turns(self) -> np.ndarray:
        return np.array([c.turns for c in self.circuits()], dtype=float)

    def total_length_m(self) -> list[float]:
        """Filament path length per circuit, summed over the coils of that circuit."""
        out = []
        for group in self.filaments:
            total = 0.0
            for xyz in group:
                total += float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
            out.append(total)
        return out


_NAMELIST_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")
_COILS_START = "** coils_dot_starts_below **"

#: Coil group names in ``coils.w7x`` in file order, mapped to circuit keys. The
#: ``AAE`` designations are the IPP winding-pack identifiers: groups 1-5 are the
#: five non-planar coil types, groups 6-7 the two planar types.
_W7X_GROUP_TO_CIRCUIT = {
    "AAE10_SC": "npc1",
    "AAE29_SC": "npc2",
    "AAE38_SC": "npc3",
    "AAE47_SC": "npc4",
    "AAE56_SC": "npc5",
    "AAE14_SC": "pca",
    "AAE23_SC": "pcb",
    # Constructed normally conducting coils, appended by
    # w7x_twin.auxiliary_coils.write_extended_coils_file.
    "TRIM_A1": "trim_a1",
    "TRIM_A2": "trim_a2",
    "TRIM_A3": "trim_a3",
    "TRIM_A4": "trim_a4",
    "TRIM_B1": "trim_b1",
    **{f"CONTROL_{m}U": f"cc{m}u" for m in range(1, NUM_FIELD_PERIODS + 1)},
    **{f"CONTROL_{m}L": f"cc{m}l" for m in range(1, NUM_FIELD_PERIODS + 1)},
}


def _parse_namelist(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("&"):
            inside = True
            continue
        if inside and stripped == "/":
            break
        if not inside or not stripped or stripped.startswith("!"):
            continue
        match = _NAMELIST_RE.match(stripped)
        if match:
            values[match.group(1).upper()] = match.group(2).strip()
    return values


def _as_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _as_bool(text: str) -> bool:
    return text.strip().strip(".").lower().startswith("t")


def load_coils(path: str | Path) -> CoilSet:
    """Parse a MAKEGRID coils file with an embedded ``&MGRID_NLI`` namelist."""
    path = Path(path)
    lines = path.read_text().splitlines()

    namelist = _parse_namelist(lines)
    start = next(i for i, l in enumerate(lines) if _COILS_START in l)
    body = lines[start + 1 :]

    periods = NUM_FIELD_PERIODS
    for line in body[:5]:
        if line.strip().lower().startswith("periods"):
            periods = int(line.split()[1])
            break

    grid = FieldGrid(
        r_min=_as_float(namelist["RMIN"]),
        r_max=_as_float(namelist["RMAX"]),
        num_r=int(namelist["IR"]),
        z_min=_as_float(namelist["ZMIN"]),
        z_max=_as_float(namelist["ZMAX"]),
        num_z=int(namelist["JZ"]),
        num_phi=int(namelist["KP"]),
        num_field_periods=periods,
        stellarator_symmetric=_as_bool(namelist.get("LSTELL_SYM", ".TRUE.")),
        normalize_by_currents=namelist.get("MGRID_MODE", "'R'").strip("'\" ").upper()
        == "S",
    )

    groups: dict[str, list[np.ndarray]] = {}
    order: list[str] = []
    currents: dict[str, float] = {}
    points: list[tuple[float, float, float]] = []
    current_of_coil = 0.0

    for line in body:
        tokens = line.split()
        if len(tokens) == 4:
            points.append((float(tokens[0]), float(tokens[1]), float(tokens[2])))
            current_of_coil = float(tokens[3])
        elif len(tokens) == 6:
            # Closing point of one coil: x y z 0.0 <group index> <group name>
            points.append((float(tokens[0]), float(tokens[1]), float(tokens[2])))
            name = tokens[5]
            if name not in groups:
                groups[name] = []
                order.append(name)
                currents[name] = current_of_coil
            groups[name].append(np.array(points, dtype=float))
            points = []
        elif tokens and tokens[0].lower() == "end":
            break

    unknown = [n for n in order if n not in _W7X_GROUP_TO_CIRCUIT]
    if unknown:
        raise ValueError(f"unrecognised coil groups in {path.name}: {unknown}")

    return CoilSet(
        path=path,
        grid=grid,
        circuit_keys=[_W7X_GROUP_TO_CIRCUIT[n] for n in order],
        group_names=order,
        filaments=[groups[n] for n in order],
        file_currents=[currents[n] for n in order],
    )


#: Order the control circuits are driven in as an n = 2 waveform steps round the machine.
CONTROL_ORDER: tuple[str, ...] = tuple(
    f"cc{module}{side}" for module in range(1, NUM_FIELD_PERIODS + 1) for side in "ul"
)


def control_waveform(
    amplitude_a: float, phase_degrees: float, mode: int = 2
) -> dict[str, float]:
    """Control currents I_k = A0 cos(2 pi mode (k - 1) / 5 - phi), upper and lower together, in A per turn."""
    phase = np.radians(phase_degrees)
    out: dict[str, float] = {}
    for index in range(NUM_FIELD_PERIODS):
        value = float(
            amplitude_a * np.cos(2.0 * np.pi * mode * index / NUM_FIELD_PERIODS - phase)
        )
        out[f"cc{index + 1}u"] = value
        out[f"cc{index + 1}l"] = value
    return out

# -- from configs -----------------------------------------------------------------

CIRCUIT_ORDER = ("npc1", "npc2", "npc3", "npc4", "npc5", "pca", "pcb")


@dataclasses.dataclass(frozen=True)
class Configuration:
    key: str
    label: str
    currents: tuple[float, ...]
    source: str
    control_coils: tuple[float, float] | None = None
    note: str = ""

    def as_extcur(self) -> np.ndarray:
        return np.array(self.currents, dtype=float)

    @property
    def ratios(self) -> np.ndarray:
        c = self.as_extcur()
        return c / c[0]

    def scaled_to(self, reference_current: float) -> np.ndarray:
        """Same current ratios, rescaled so the first non-planar circuit matches."""
        return self.as_extcur() * (reference_current / self.currents[0])


_IPP_REF = (
    "IPP VMEC reference run input.w7x_ref_167_12_12, distributed with "
    "indata2json and used by the VMEC++ free-boundary W7-X example"
)
_ORNL = (
    "ORNL-Fusion/util-library make_poincare_plots_OP12a_configs.m, tapers used "
    "for W7-X OP1.2a and OP2 modelling (currents per winding)"
)

#: Configurations whose currents come directly from a published or distributed source.
MEASURED: dict[str, Configuration] = {
    "standard": Configuration(
        key="standard",
        label="Standard (EIM-type)",
        currents=(12883.0, 12883.0, 12883.0, 12883.0, 12883.0, 0.0, 0.0),
        source=_ORNL,
        note=(
            "Defining property of the standard configuration: equal current in all "
            "five modular coil types and no planar coil current."
        ),
    ),
    "high_mirror_ref167": Configuration(
        key="high_mirror_ref167",
        label="High mirror (IPP reference 167)",
        currents=(13000.0, 13260.0, 14040.0, 12090.0, 10959.0, 0.0, 0.0),
        source=_IPP_REF,
        note="Ratios 1 : 1.020 : 1.080 : 0.930 : 0.843, planar coils off.",
    ),
    "narrow_mirror": Configuration(
        key="narrow_mirror",
        label="Narrow mirror",
        currents=tuple(
            12556.0 * r for r in (1.0, 1.02, 1.08, 0.97, 0.88, 0.15, -0.15)
        ),
        source=_ORNL,
    ),
    "op2_22ka": Configuration(
        key="op2_22ka",
        label="OP2 reference, 22 kA",
        currents=(12130.0, 12000.0, 12255.0, 13541.0, 13635.0, 9000.0, -2900.0),
        source=_ORNL,
    ),
    "op12a_0ka_mimic": Configuration(
        key="op12a_0ka_mimic",
        label="OP1.2a 0 kA mimic (EES+252)",
        currents=(12022.0, 11897.0, 12148.0, 13399.0, 13524.0, 8219.0, -3005.0),
        control_coils=(2500.0, -2500.0),
        source=_ORNL,
    ),
    "op12a_11ka_mimic": Configuration(
        key="op12a_11ka_mimic",
        label="OP1.2a 11 kA mimic (EFS+252)",
        currents=(12129.0, 12002.0, 12255.0, 13519.0, 13645.0, 7391.0, -3980.0),
        control_coils=(2500.0, -2500.0),
        source=_ORNL,
    ),
    "op12a_22ka_mimic": Configuration(
        key="op12a_22ka_mimic",
        label="OP1.2a 22 kA mimic (EGS+252)",
        currents=(12243.0, 12116.0, 12371.0, 13646.0, 13774.0, 6504.0, -4973.0),
        control_coils=(2500.0, -2500.0),
        source=_ORNL,
    ),
    "op12a_32ka_mimic": Configuration(
        key="op12a_32ka_mimic",
        label="OP1.2a 32 kA mimic (EGS001+252)",
        currents=(12359.0, 12230.0, 12487.0, 13775.0, 13904.0, 5600.0, -5987.0),
        control_coils=(2500.0, -2500.0),
        source=_ORNL,
    ),
    "op12a_43ka_mimic": Configuration(
        key="op12a_43ka_mimic",
        label="OP1.2a 43 kA mimic (FHS+252)",
        currents=(12477.0, 12347.0, 12607.0, 13907.0, 14037.0, 4679.0, -7019.0),
        control_coils=(2500.0, -2500.0),
        source=_ORNL,
    ),
}

#: Filled in by :func:`build_divertor_scenarios` once the planar current for a target
#: edge transform has been solved for; kept separate so computed and sourced numbers
#: are never confused.
DERIVED: dict[str, Configuration] = {}


def get(key: str) -> Configuration:
    if key in MEASURED:
        return MEASURED[key]
    if key in DERIVED:
        return DERIVED[key]
    raise KeyError(f"unknown configuration {key!r}; known: {sorted(all_keys())}")


def all_keys() -> list[str]:
    return sorted(set(MEASURED) | set(DERIVED))


def with_planar(
    base: Configuration, planar_current: float, key: str, label: str, source: str
) -> Configuration:
    """Same modular currents as ``base``, both planar circuits at ``planar_current``."""
    modular = base.currents[:5]
    return Configuration(
        key=key,
        label=label,
        currents=(*modular, planar_current, planar_current),
        source=source,
    )

# -- from configure ---------------------------------------------------------------

@dataclasses.dataclass
class SolveTrace:
    """One inverse solve: the actuator values tried and the transforms they gave."""

    planar_currents: list[float]
    edge_transforms: list[float]
    converged: bool
    residual: float

    def as_rows(self) -> list[tuple[float, float]]:
        return list(zip(self.planar_currents, self.edge_transforms, strict=True))


def edge_transform(
    twin: "Twin",
    modular: tuple[float, ...],
    planar_current: float,
    resolution: "Resolution | None" = None,
) -> float:
    """Edge rotational transform for the given modular and planar currents."""
    from w7x_twin.mhd import diagnostics
    from w7x_twin.mhd.equilibrium import MachineState, SCAN, Twin

    resolution = resolution or SCAN


    currents = np.array([*modular, planar_current, planar_current], dtype=float)
    state = MachineState(
        currents=currents,
        toroidal_flux_wb=twin.toroidal_flux_for(currents),
        label=f"planar={planar_current:.1f}A",
    )
    output = twin.solve(state, resolution)
    return float(np.asarray(output.wout.iotaf)[-1])


def solve_planar_for_edge_transform(
    twin: "Twin",
    target: float,
    base: "Configuration | str" = "standard",
    resolution: "Resolution | None" = None,
    tolerance: float = 2e-4,
    max_iterations: int = 12,
    bracket: tuple[float, float] = (-12000.0, 12000.0),
    verbose: bool = True,
) -> tuple[float, SolveTrace]:
    """Planar current (both circuits together) giving edge transform ``target``, with the search trace."""
    from w7x_twin.mhd import diagnostics
    from w7x_twin.mhd.equilibrium import MachineState, SCAN, Twin

    resolution = resolution or SCAN

    if isinstance(base, str):
        base = get(base)
    modular = tuple(
        float(x) * (13000.0 / base.currents[0]) for x in base.currents[:5]
    )

    lo, hi = bracket
    tried: list[float] = []
    got: list[float] = []

    def evaluate(current: float) -> float:
        value = edge_transform(twin, modular, current, resolution)
        tried.append(current)
        got.append(value)
        if verbose:
            print(f"    planar {current:9.1f} A/turn -> iota_edge {value:.5f}")
        return value

    f_lo = evaluate(lo) - target
    f_hi = evaluate(hi) - target
    if f_lo * f_hi > 0:
        return float("nan"), SolveTrace(tried, got, False, min(abs(f_lo), abs(f_hi)))

    current, residual = float("nan"), float("inf")
    for _ in range(max_iterations):
        # Secant step, kept inside the bracket.
        denominator = f_hi - f_lo
        current = (
            0.5 * (lo + hi)
            if abs(denominator) < 1e-12
            else hi - f_hi * (hi - lo) / denominator
        )
        if not (min(lo, hi) < current < max(lo, hi)):
            current = 0.5 * (lo + hi)

        residual = evaluate(current) - target
        if abs(residual) < tolerance:
            return current, SolveTrace(tried, got, True, abs(residual))
        if f_lo * residual < 0:
            hi, f_hi = current, residual
        else:
            lo, f_lo = current, residual

    return current, SolveTrace(tried, got, False, abs(residual))


#: Divertor scenarios named by the island chain that carries the strike lines.
DIVERTOR_SCENARIOS: dict[str, tuple[float, str]] = {
    "low_iota": (5.0 / 6.0, "Low iota, edge transform at the 5/6 island chain"),
    "standard_iota": (1.0, "Standard, edge transform at the 5/5 island chain"),
    "high_iota": (5.0 / 4.0, "High iota, edge transform at the 5/4 island chain"),
}


def build_divertor_scenarios(
    twin: "Twin", resolution: "Resolution | None" = None, verbose: bool = True
) -> dict[str, Configuration]:
    """Solve and register the low, standard and high transform scenarios."""
    from w7x_twin.mhd import diagnostics
    from w7x_twin.mhd.equilibrium import MachineState, SCAN, Twin

    resolution = resolution or SCAN

    out: dict[str, Configuration] = {}
    for key, (target, label) in DIVERTOR_SCENARIOS.items():
        if verbose:
            print(f"[configure] {key}: target edge transform {target:.5f}")
        current, trace = solve_planar_for_edge_transform(
            twin, target, resolution=resolution, verbose=verbose
        )
        if not trace.converged:
            if verbose:
                print(f"[configure] {key}: no bracket, residual {trace.residual:.4f}")
            continue
        config = with_planar(
            get("standard"),
            current,
            key=key,
            label=label,
            source=(
                "derived: planar coil current solved by the twin so the edge "
                f"rotational transform reaches {target:.5f}"
            ),
        )
        DERIVED[key] = config
        out[key] = config
        if verbose:
            print(f"[configure] {key}: planar {current:.1f} A/turn\n")
    return out

# -- from provenance ---------------------------------------------------------------

# Per-part content hashes of the geometry, so each consumer keys on what it reads.

#: Digest length carried in cache keys and printed as the version.
DIGEST_CHARS = 12


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:DIGEST_CHARS]


def file_digest(path: str | Path) -> str:
    """Digest of one file's contents, or ``absent`` when it is not there."""
    path = Path(path)
    return _digest(path.read_bytes()) if path.is_file() else "absent"


def directory_digest(directory: str | Path, pattern: str = "*") -> str:
    """Digest over a directory's matching files, by sorted name and contents."""
    directory = Path(directory)
    if not directory.is_dir():
        return "absent"
    accumulator = hashlib.sha256()
    for path in sorted(p for p in directory.glob(pattern) if p.is_file()):
        accumulator.update(path.name.encode())
        accumulator.update(path.read_bytes())
    return accumulator.hexdigest()[:DIGEST_CHARS]


def values_digest(values: dict) -> str:
    """Digest of a set of scalar parameters, such as a field grid."""
    return _digest(json.dumps(values, sort_keys=True, default=str).encode())


@dataclasses.dataclass(frozen=True)
class Epoch:
    """One in-vessel configuration: limiter, test divertor unit, or actively cooled divertor."""

    key: str
    label: str
    campaigns: tuple[str, ...]
    #: Components the epoch presents to the plasma, by the names walls.py uses.
    components: tuple[str, ...]
    cooled: bool


#: The in-vessel configurations, in the order the machine ran them.
EPOCHS: tuple[Epoch, ...] = (
    Epoch(
        key="limiter",
        label="Inertially cooled graphite limiter",
        campaigns=("OP1.1",),
        components=(),
        cooled=False,
    ),
    Epoch(
        key="tdu",
        label="Inertially cooled test divertor unit",
        campaigns=("OP1.2a", "OP1.2b"),
        components=(
            "divertor horizontal target, upper",
            "divertor vertical target, upper",
            "divertor horizontal target, lower",
            "baffle, horizontal upper",
            "baffle, horizontal lower",
            "baffle, horizontal mid",
            "baffle, vertical upper",
            "baffle, vertical lower",
            "baffle, vertical n8",
        ),
        cooled=False,
    ),
    Epoch(
        key="hhf",
        label="Actively cooled high heat flux divertor",
        campaigns=("OP2.1", "OP2.2", "OP2.3", "OP2.4", "OP2.5"),
        components=(
            "divertor horizontal target, upper",
            "divertor vertical target, upper",
            "divertor horizontal target, lower",
            "baffle, horizontal upper",
            "baffle, horizontal lower",
            "baffle, horizontal mid",
            "baffle, vertical upper",
            "baffle, vertical lower",
            "baffle, vertical n8",
            "scraper element",
        ),
        cooled=True,
    ),
)

#: The epoch a result belongs to unless one is named.
DEFAULT_EPOCH = "hhf"


def epoch(key: str = DEFAULT_EPOCH) -> Epoch:
    known = {item.key: item for item in EPOCHS}
    if key not in known:
        raise KeyError(f"unknown epoch {key!r}; have {sorted(known)}")
    return known[key]


def epoch_of_campaign(campaign: str) -> Epoch:
    """The epoch an operation campaign ran in, so a programme identifier resolves one."""
    for item in EPOCHS:
        if campaign in item.campaigns:
            return item
    raise KeyError(f"no epoch carries campaign {campaign!r}")


@dataclasses.dataclass(frozen=True)
class GeometryVersion:
    """Per-input digests of the geometry, and one digest over all of them."""

    parts: tuple[tuple[str, str], ...]

    @property
    def digest(self) -> str:
        return _digest(
            "".join(f"{name}:{value};" for name, value in self.parts).encode()
        )

    def subset(self, *names: str) -> str:
        """Digest over the named parts only, for a consumer that reads just those."""
        lookup = dict(self.parts)
        missing = [name for name in names if name not in lookup]
        if missing:
            raise KeyError(f"no geometry part {missing}; have {sorted(lookup)}")
        return _digest("".join(f"{name}:{lookup[name]};" for name in names).encode())

    def as_dict(self) -> dict[str, str]:
        return {"geometry": self.digest, **dict(self.parts)}

    def __str__(self) -> str:
        parts = " ".join(f"{name}={value}" for name, value in self.parts)
        return f"geometry {self.digest} [{parts}]"


def geometry_version(
    coils_path: str | Path,
    grid_parameters: dict | None = None,
    template_path: str | Path | None = None,
    vessel_path: str | Path | None = None,
    component_dir: str | Path | None = None,
    epoch_key: str = DEFAULT_EPOCH,
) -> GeometryVersion:
    """Geometry version over the files and parameters in use, the epoch as its own part."""
    from w7x_twin.hardware import coils as constructed

    era = epoch(epoch_key)
    # The constructed trim and control coils enter as their own part, so the field
    # tables keyed on the coils file alone stay valid across pack refinements while
    # anything reading the constructed set moves with them.
    parts: list[tuple[str, str]] = [
        ("epoch", era.key),
        ("coils", file_digest(coils_path)),
        ("constructed", constructed.constructed_digest()),
    ]
    if grid_parameters is not None:
        parts.append(("grid", values_digest(grid_parameters)))
    if template_path is not None:
        parts.append(("template", file_digest(template_path)))
    if vessel_path is not None:
        parts.append(("vessel", file_digest(vessel_path)))
    if component_dir is not None:
        parts.append(
            (
                "components",
                _digest(
                    (
                        directory_digest(component_dir)
                        + "|"
                        + ",".join(era.components)
                    ).encode()
                ),
            )
        )
    return GeometryVersion(parts=tuple(parts))

# -- from machine_state ------------------------------------------------------------

# Deposited energy per plasma-facing element accumulated across programmes, keyed by
# element, module, unit and the geometry it was computed under.

#: Energy below which an entry is not worth carrying, in joules.
NEGLIGIBLE_J = 1.0


@dataclasses.dataclass
class ElementLoad:
    """What one divertor or baffle element has taken."""

    element: str
    module: int
    unit: str
    energy_j: float = 0.0
    peak_flux_w_m2: float = 0.0
    seconds: float = 0.0
    programmes: int = 0

    @property
    def key(self) -> str:
        return f"{self.element}|{self.module}|{self.unit}"

    def add(self, energy_j: float, flux_w_m2: float, seconds: float) -> None:
        self.energy_j += float(energy_j)
        self.peak_flux_w_m2 = max(self.peak_flux_w_m2, float(flux_w_m2))
        self.seconds += float(seconds)
        self.programmes += 1


@dataclasses.dataclass
class Programme:
    """One discharge, as far as the deposition needs to know about it."""

    identifier: str
    configuration: str
    heating_power_w: float
    duration_s: float
    carbon_fraction: float = 0.0

    @property
    def energy_j(self) -> float:
        return self.heating_power_w * self.duration_s


@dataclasses.dataclass
class CampaignLoad:
    """Accumulated load over a set of programmes, and the geometry it was computed on."""

    geometry: dict
    epoch: str
    loads: dict[str, ElementLoad] = dataclasses.field(default_factory=dict)
    programmes: list[dict] = dataclasses.field(default_factory=list)

    def entry(self, element: str, module: int, unit: str) -> ElementLoad:
        key = f"{element}|{module}|{unit}"
        if key not in self.loads:
            self.loads[key] = ElementLoad(element=element, module=module, unit=unit)
        return self.loads[key]

    def add_programme(
        self,
        programme: Programme,
        deposition: dict[tuple[str, int, str], tuple[float, float]],
    ) -> None:
        """Add one programme's deposition, given ``{(element, module, unit): (W, W/m^2)}``."""
        for (element, module, unit), (power, flux) in deposition.items():
            energy = power * programme.duration_s
            if energy < NEGLIGIBLE_J:
                continue
            self.entry(element, module, unit).add(energy, flux, programme.duration_s)
        self.programmes.append(
            {**dataclasses.asdict(programme), "energy_j": programme.energy_j}
        )

    def ranked(self) -> list[ElementLoad]:
        return sorted(self.loads.values(), key=lambda load: -load.energy_j)

    def total_energy_j(self) -> float:
        return sum(load.energy_j for load in self.loads.values())

    def module_spread(self) -> float:
        """Most-to-least loaded module ratio at fixed element and unit: the periodicity test."""
        groups: dict[tuple[str, str], list[float]] = {}
        for load in self.loads.values():
            groups.setdefault((load.element, load.unit), []).append(load.energy_j)
        worst = 1.0
        for values in groups.values():
            low = min(values)
            if low > 0:
                worst = max(worst, max(values) / low)
        return worst

    def unit_ratio(self) -> dict[str, float]:
        """Upper-to-lower energy ratio of each element, summed over the modules."""
        upper: dict[str, float] = {}
        lower: dict[str, float] = {}
        for load in self.loads.values():
            target = upper if load.unit == "upper" else lower
            target[load.element] = target.get(load.element, 0.0) + load.energy_j
        return {
            element: upper.get(element, 0.0) / value
            for element, value in lower.items()
            if value > 0
        }

    def as_dict(self) -> dict:
        return {
            "geometry": self.geometry,
            "epoch": self.epoch,
            "total_energy_j": self.total_energy_j(),
            "module_spread": self.module_spread(),
            "upper_over_lower": self.unit_ratio(),
            "programmes": self.programmes,
            "loads": [dataclasses.asdict(load) for load in self.ranked()],
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path

    @staticmethod
    def read(path: str | Path, geometry: dict, epoch: str) -> "CampaignLoad":
        """Read an existing record, or start one; another geometry's record is not continued."""
        path = Path(path)
        if not path.exists():
            return CampaignLoad(geometry=geometry, epoch=epoch)
        stored = json.loads(path.read_text())
        if stored.get("geometry", {}).get("geometry") != geometry.get("geometry"):
            raise ValueError(
                f"{path} carries geometry {stored.get('geometry', {}).get('geometry')}, "
                f"not {geometry.get('geometry')}; the elements it names are not these"
            )
        if stored.get("epoch") != epoch:
            raise ValueError(
                f"{path} carries epoch {stored.get('epoch')}, not {epoch}"
            )
        record = CampaignLoad(
            geometry=stored["geometry"],
            epoch=stored["epoch"],
            programmes=stored.get("programmes", []),
        )
        for entry in stored.get("loads", []):
            load = ElementLoad(**entry)
            record.loads[load.key] = load
        return record
