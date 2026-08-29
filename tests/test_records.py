"""Guards on what the records claim, and on what the package needs to be importable."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "w7x_twin"


# -- the solver stays out of module scope -----------------------------------

#: Distributions that exist for one platform, so a module importing either at its
#: own scope puts everything in it behind that platform.
PLATFORM_BOUND = {"vmecpp", "simsopt"}


def module_scope_imports(path: Path) -> set[str]:
    """Distributions a module imports at its own scope, TYPE_CHECKING blocks aside."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_no_module_imports_the_solver_at_its_own_scope():
    """Reading a record or tracing a field must not need a solver that builds on one OS.

    Every function that reaches VMEC++ is handed an equilibrium or builds one, and
    imports it there. Hoisting that to module scope makes the whole package, and with
    it the whole test suite, uncollectable wherever the extension does not build.
    """
    offenders = []
    for path in sorted(SOURCE.rglob("*.py")):
        bound = module_scope_imports(path) & PLATFORM_BOUND
        if bound:
            offenders.append(f"{path.relative_to(ROOT).as_posix()}: {sorted(bound)}")
    assert not offenders, "platform-bound imports at module scope: " + "; ".join(offenders)


def test_every_module_of_the_package_imports():
    """The guard above is structural; this one is that the package loads."""
    import importlib
    import pkgutil

    import w7x_twin

    #: ``__main__`` dispatches and exits on import, and the two workers run in another
    #: interpreter and import what only that one carries.
    SKIP = {
        "w7x_twin.__main__",
        "w7x_twin.magnetics._biot_savart_gpu",
        "w7x_twin.plasma._effective_ripple_desc",
    }
    failed = []
    for info in pkgutil.walk_packages(w7x_twin.__path__, "w7x_twin."):
        if info.name in SKIP:
            continue
        try:
            importlib.import_module(info.name)
        except ImportError as error:
            failed.append(f"{info.name}: {error}")
    assert not failed, "modules that will not import: " + "; ".join(failed)


# -- the constructed windings -----------------------------------------------

def test_the_control_saddle_carries_no_repeated_vertex():
    """A repeated vertex has a zero tangent, which carries no winding frame."""
    from w7x_twin.hardware import coils as coil_geometry
    from w7x_twin.hardware import walls

    part = ROOT / "data" / "vessel.part"
    if not part.is_file():
        pytest.skip("vessel contour not present")
    vessel = walls.load_vessel(part)

    for group in coil_geometry.control_coils(vessel):
        for filament in group.filaments:
            steps = np.linalg.norm(np.diff(filament, axis=0), axis=1)
            # The closing step is the only one allowed to vanish.
            assert np.all(steps[:-1] > 0.0), f"{group.key} repeats a vertex"


def test_every_constructed_circuit_carries_its_declared_turns():
    """One filament per conductor turn, on the trim coils and the control coils alike."""
    from w7x_twin.hardware import coils as coil_geometry
    from w7x_twin.hardware import machine, walls

    part = ROOT / "data" / "vessel.part"
    if not part.is_file():
        pytest.skip("vessel contour not present")
    vessel = walls.load_vessel(part)

    declared = {c.key: c.turns for c in machine.AUXILIARY_CIRCUITS}
    built = coil_geometry.trim_coils() + coil_geometry.control_coils(vessel)
    assert {g.key for g in built} == set(declared)
    for group in built:
        assert len(group.filaments) == declared[group.key], group.key


# -- what the documents say the records say ---------------------------------

def test_the_readme_states_the_geometry_the_inputs_produce():
    """The printed version and the inputs behind it drift apart unless a check reads both."""
    from w7x_twin.analyses import _common

    if not (ROOT / "data" / "coils.w7x").is_file():
        pytest.skip("machine description not fetched")
    version = _common.current_geometry()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stated = re.findall(r"^geometry ([0-9a-f]{12}) \[", readme, re.M)
    assert stated, "the README no longer prints a geometry version"
    assert stated[0] == version.digest, (
        f"the README prints geometry {stated[0]}; the inputs give {version.digest}"
    )
    for part, value in version.parts:
        assert f"{part}={value}" in readme, f"the README does not print {part}={value}"


def test_the_epoch_table_states_the_epochs_digests():
    from w7x_twin.analyses import _common
    from w7x_twin.hardware import machine

    if not (ROOT / "data" / "coils.w7x").is_file():
        pytest.skip("machine description not fetched")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for era in machine.EPOCHS:
        digest = _common.current_geometry(era.key).digest
        assert re.search(rf"^{era.key}\s+{digest}\s", readme, re.M), (
            f"the README's epoch table does not give {era.key} as {digest}"
        )


def test_the_physics_record_states_the_number_of_checks_that_agree():
    """The count is written out in words, so it drifts silently as checks are added."""
    stored = json.loads((ROOT / "results" / "validation.json").read_text())
    words = {
        30: "thirty", 31: "thirty-one", 32: "thirty-two", 33: "thirty-three",
        34: "thirty-four", 35: "thirty-five", 36: "thirty-six", 37: "thirty-seven",
        38: "thirty-eight", 39: "thirty-nine", 40: "forty",
    }
    total = int(stored["total"])
    assert stored["agreed"] == total, "the stored record does not agree with itself"
    assert total in words, f"{total} checks; extend the table in this test"
    physics = (ROOT / "docs" / "physics.md").read_text(encoding="utf-8")
    assert f"All {words[total]} agree." in physics, (
        f"docs/physics.md does not say all {words[total]} agree"
    )


# -- the audit's own tables against the code they describe ------------------

def test_the_audit_names_every_record_it_walks():
    from w7x_twin.analyses import data

    walked = sorted(
        path.as_posix() for path in (ROOT / "results").rglob("*.json")
        if path.relative_to(ROOT).as_posix() not in data.EXPORT_BUNDLES
    )
    unnamed = [
        Path(name).relative_to("").as_posix() for name in walked
        if Path(name).relative_to(ROOT).as_posix() not in data.WRITTEN_BY
        and Path(name).relative_to(ROOT).as_posix() not in data.UNSTAMPED
    ]
    assert not unnamed, (
        "records the audit cannot name a command for: "
        + "; ".join(Path(n).name for n in unnamed)
    )


def test_the_audit_declares_the_dependencies_the_analyses_stamp():
    """`DECLARES_READS` is what tells a missing stamp from a record that reads nothing."""
    from w7x_twin.analyses import data

    stamped: set[str] = set()
    for path in sorted((SOURCE / "analyses").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            named = getattr(node.func, "attr", getattr(node.func, "id", None))
            if named != "write_record":
                continue
            if any(keyword.arg == "reads" for keyword in node.keywords):
                stamped.add(path.stem)
    declared = {Path(name).name for name in data.DECLARES_READS}
    assert stamped, "no analysis stamps its inputs any more"
    assert declared, "the audit declares no dependencies"
    # Every declared record must exist; a renamed record would otherwise be audited
    # as though it read nothing.
    for name in data.DECLARES_READS:
        assert (ROOT / name).is_file(), f"{name} is declared but absent"
        for source in data.DECLARES_READS[name]:
            assert (ROOT / source).is_file(), f"{name} declares absent input {source}"


def test_records_that_carry_input_digests_carry_current_ones():
    """The same check `w7x-twin records` makes, so a stale input fails the suite too."""
    from w7x_twin.analyses import _common

    stale = []
    for path in sorted((ROOT / "results").rglob("*.json")):
        try:
            stored = json.loads(path.read_text())
        except ValueError:
            continue
        reads = stored.get("reads") if isinstance(stored, dict) else None
        if not isinstance(reads, dict):
            continue
        for source, digest in reads.items():
            now = _common.file_digest(ROOT / source)
            if now != digest:
                stale.append(f"{path.name} read {Path(source).name} at {digest}, now {now}")
    assert not stale, "records standing on inputs that have moved: " + "; ".join(stale)
