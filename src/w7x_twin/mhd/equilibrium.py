"""Free-boundary equilibria, and the Boozer coordinates derived from them."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import numpy as np
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING

from w7x_twin.hardware import machine
from w7x_twin.magnetics import field

# Twin's methods import the solver where they call it, so the scenario, resolution
# and machine-state records here are readable without it installed.
if TYPE_CHECKING:
    import vmecpp


# -- from equilibrium -------------------------------------------------------------

REFERENCE_TOROIDAL_FLUX_WB = 1.740

#: Modular coil current of the IPP reference case, per turn in amperes.
REFERENCE_MODULAR_CURRENT_A = 13000.0

#: Major radius at which the toroidal field direction is sampled.
FIELD_SIGN_PROBE_R_M = 5.93


@dataclasses.dataclass
class Scenario:
    """Plasma state the coils do not set: the ``am`` power-series shape at unit leading
    coefficient, its level ``peak_pressure_pa``, or a spline, and the net current."""

    peak_pressure_pa: float = 0.0
    pressure_profile: tuple[float, ...] = (1.0, -1.0)
    net_toroidal_current_a: float = 0.0
    current_profile: tuple[float, ...] = (1.0,)

    #: Optional (s, p) knots. When set, the pressure is a cubic spline through them
    #: in pascals and the power series above is unused.
    pressure_spline: tuple[np.ndarray, np.ndarray] | None = None
    #: Optional (s, dI/ds) knots for the enclosed toroidal current profile.
    current_spline: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def is_vacuum(self) -> bool:
        return self.peak_pressure_pa == 0.0 and self.net_toroidal_current_a == 0.0

    @staticmethod
    def from_pressure_spline(
        knots_s: np.ndarray, knots_p: np.ndarray
    ) -> Scenario:
        """Pressure as a spline through (s, Pa) knots and no net current."""
        return Scenario(
            pressure_spline=(knots_s, knots_p),
            peak_pressure_pa=1.0,
            pressure_profile=(1.0,),
        )


@dataclasses.dataclass
class MachineState:
    """A complete actuator setting of the machine."""

    currents: np.ndarray
    toroidal_flux_wb: float
    scenario: Scenario = dataclasses.field(default_factory=Scenario)
    label: str = ""

    @staticmethod
    def from_configuration(
        config: machine.Configuration | str,
        field_scale: float = 1.0,
        scenario: Scenario | None = None,
    ) -> MachineState:
        """Actuator setting for a named configuration; ``field_scale`` one is 13 kA per turn."""
        if isinstance(config, str):
            config = machine.get(config)
        currents = config.scaled_to(REFERENCE_MODULAR_CURRENT_A * field_scale)
        return MachineState(
            currents=currents,
            toroidal_flux_wb=REFERENCE_TOROIDAL_FLUX_WB * field_scale,
            scenario=scenario or Scenario(),
            label=config.label,
        )

    def digest(self) -> str:
        def spline(entry):
            return None if entry is None else [list(map(float, a)) for a in entry]

        payload = {
            "currents": [float(x) for x in self.currents],
            "flux": self.toroidal_flux_wb,
            "peak_pressure": self.scenario.peak_pressure_pa,
            "pressure_profile": list(self.scenario.pressure_profile),
            "curtor": self.scenario.net_toroidal_current_a,
            "current_profile": list(self.scenario.current_profile),
            "pressure_spline": spline(self.scenario.pressure_spline),
            "current_spline": spline(self.scenario.current_spline),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class Resolution:
    """Fourier and radial resolution of the solve."""

    mpol: int
    ntor: int
    ns: tuple[int, ...]
    ftol: tuple[float, ...]
    niter: int = 25000

    def digest(self) -> str:
        return f"m{self.mpol}n{self.ntor}s{'-'.join(map(str, self.ns))}"

    def final_stage(self) -> Resolution:
        """The last multigrid stage on its own, which is what a hot restart needs."""
        return dataclasses.replace(
            self, ns=(self.ns[-1],), ftol=(self.ftol[-1],)
        )


#: Cheap enough for scans; resolves the transform and the mirror well.
SCAN = Resolution(mpol=7, ntor=6, ns=(25, 51), ftol=(1e-8, 1e-11))
#: Matches the Fourier resolution of the shipped W7-X free-boundary case.
REFERENCE = Resolution(mpol=8, ntor=8, ns=(25, 51, 99), ftol=(1e-8, 1e-10, 1e-12))
#: Full resolution of the IPP reference runs.
HIGH = Resolution(mpol=12, ntor=12, ns=(25, 51, 99), ftol=(1e-8, 1e-10, 1e-12))


class Twin:
    """The W7-X forward model."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        cache_dir: str | Path = "cache",
        coils_file: str = "coils.w7x",
        template: str = "w7x_free_bdy_vac.json",
        verbose: bool = True,
        epoch: str = machine.DEFAULT_EPOCH,
    ) -> None:
        import vmecpp

        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.coils = machine.load_coils(self.data_dir / coils_file)
        self.response = field.build_response_table(
            self.coils, cache_dir=self.cache_dir, verbose=verbose
        )
        self._template = vmecpp.VmecInput.from_file(self.data_dir / template)

        # One version over every geometry input, with the parts kept apart so each
        # consumer keys on what it reads. The equilibrium depends on the coil set and
        # the field grid and not on the vessel or the components, so its cache key is
        # that subset; without it two coil models sharing an actuator setting collide,
        # which is how a finite-build solve once returned the single-filament result.
        self.epoch = machine.epoch(epoch)
        self.geometry = machine.geometry_version(
            coils_path=self.coils.path,
            grid_parameters=dataclasses.asdict(self.coils.grid),
            template_path=self.data_dir / template,
            vessel_path=self.data_dir / "vessel.part",
            component_dir=self.data_dir / "pfc",
            epoch_key=epoch,
        )
        self._field_digest = self.geometry.subset("coils", "grid")
        if verbose:
            print(f"[twin] {self.geometry}")

    def full_torus_response(self):
        """Whole-torus per-circuit response, for waveforms a per-period table would fold away."""
        if getattr(self, "_full_torus", None) is None:
            self._full_torus = field.build_response_table(
                self.coils,
                grid=field.full_torus_grid(self.coils.grid),
                cache_dir=self.cache_dir,
                verbose=self.verbose,
            )
        return self._full_torus

    # -- field conventions -------------------------------------------------
    def toroidal_field_sign(self, currents: np.ndarray) -> float:
        """Sign of the vacuum toroidal field for these circuit currents."""
        _, b_phi, _ = field.field_at(
            self.response, currents, FIELD_SIGN_PROBE_R_M, 0.0, 0.0
        )
        return float(np.sign(b_phi))

    def toroidal_flux_for(
        self, currents: np.ndarray, magnitude_wb: float = REFERENCE_TOROIDAL_FLUX_WB
    ) -> float:
        """Signed toroidal flux consistent with the field these currents produce."""
        return self.toroidal_field_sign(currents) * abs(magnitude_wb)

    def state(
        self,
        config: machine.Configuration | str,
        field_scale: float = 1.0,
        scenario: Scenario | None = None,
    ) -> MachineState:
        """Build an actuator setting with the flux sign matched to the coil field."""
        state = MachineState.from_configuration(config, field_scale, scenario)
        # Configurations name the superconducting circuits; any further circuits the
        # loaded coils file provides start unpowered.
        if len(state.currents) < self.coils.num_circuits:
            padded = np.zeros(self.coils.num_circuits)
            padded[: len(state.currents)] = state.currents
            state.currents = padded
        state.toroidal_flux_wb = self.toroidal_flux_for(
            state.currents, REFERENCE_TOROIDAL_FLUX_WB * field_scale
        )
        return state

    def with_currents(self, state: MachineState, **circuits: float) -> MachineState:
        """Copy of ``state`` with named circuits set, e.g. ``trim_a1=1800``."""
        currents = np.array(state.currents, dtype=float)
        keys = self.coils.circuit_keys
        for name, value in circuits.items():
            if name not in keys:
                raise KeyError(f"no circuit {name!r}; have {keys}")
            currents[keys.index(name)] = value
        return dataclasses.replace(
            state,
            currents=currents,
            label=f"{state.label} + " + ", ".join(f"{k}={v:g}A" for k, v in circuits.items()),
        )

    # -- input assembly ----------------------------------------------------
    def build_input(
        self, state: MachineState, resolution: Resolution
    ) -> vmecpp.VmecInput:
        import vmecpp

        vmec_input = copy.deepcopy(self._template)

        if len(state.currents) != self.coils.num_circuits:
            raise ValueError(
                f"expected {self.coils.num_circuits} circuit currents, "
                f"got {len(state.currents)}"
            )
        vmec_input.extcur = np.asarray(state.currents, dtype=float)
        vmec_input.phiedge = float(state.toroidal_flux_wb)
        vmec_input.lfreeb = True
        vmec_input.free_boundary_method = vmecpp.FreeBoundaryMethod.NESTOR

        scenario = state.scenario
        vmec_input.gamma = 0.0
        if scenario.pressure_spline is not None:
            knots_s, knots_p = scenario.pressure_spline
            vmec_input.pmass_type = "cubic_spline"
            vmec_input.am_aux_s = np.asarray(knots_s, dtype=float)
            vmec_input.am_aux_f = np.asarray(knots_p, dtype=float)
            vmec_input.am = np.zeros(1)
            vmec_input.pres_scale = 1.0
        else:
            vmec_input.pmass_type = "power_series"
            vmec_input.am = np.asarray(scenario.pressure_profile, dtype=float)
            vmec_input.pres_scale = float(scenario.peak_pressure_pa)

        vmec_input.ncurr = 1
        vmec_input.curtor = float(scenario.net_toroidal_current_a)
        if scenario.current_spline is not None:
            knots_s, knots_dids = scenario.current_spline
            vmec_input.pcurr_type = "cubic_spline_ip"
            vmec_input.ac_aux_s = np.asarray(knots_s, dtype=float)
            vmec_input.ac_aux_f = np.asarray(knots_dids, dtype=float)
            vmec_input.ac = np.zeros(1)
        else:
            vmec_input.pcurr_type = "power_series"
            vmec_input.ac = np.asarray(scenario.current_profile, dtype=float)

        vmec_input.mpol = resolution.mpol
        vmec_input.ntor = resolution.ntor
        vmec_input.ns_array = np.array(resolution.ns, dtype=np.int32)
        vmec_input.ftol_array = np.array(resolution.ftol, dtype=float)
        vmec_input.niter_array = np.full(
            len(resolution.ns), resolution.niter, dtype=np.int32
        )
        vmec_input.nzeta = max(36, 2 * resolution.ntor + 4)

        # The template boundary is stored at its own resolution; pad or truncate.
        vmec_input = _resize_boundary(vmec_input, resolution.mpol, resolution.ntor)
        return vmec_input

    # -- solve -------------------------------------------------------------
    def solve(
        self,
        state: MachineState,
        resolution: Resolution = SCAN,
        restart_from: vmecpp.VmecOutput | None = None,
        cache: bool = True,
    ) -> vmecpp.VmecOutput:
        import vmecpp

        key = f"eq_{self._field_digest}_{state.digest()}_{resolution.digest()}"
        path = self.cache_dir / f"{key}.pkl"
        if cache and path.exists():
            if self.verbose:
                print(f"[twin] cache hit {path.name}")
            with path.open("rb") as handle:
                return pickle.load(handle)

        # A hot restart continues an existing radial grid, so the solve runs as a
        # single stage at the resolution the restart state was converged on.
        solve_resolution = (
            resolution.final_stage() if restart_from is not None else resolution
        )
        vmec_input = self.build_input(state, solve_resolution)
        if self.verbose:
            print(
                f"[twin] solving {state.label or 'custom'}: "
                f"extcur={np.array2string(np.asarray(state.currents), precision=0)} "
                f"phiedge={state.toroidal_flux_wb:.4f} "
                f"p0={state.scenario.peak_pressure_pa:.4g} Pa "
                f"Itor={state.scenario.net_toroidal_current_a:.4g} A "
                f"[{resolution.digest()}]"
            )
        started = time.monotonic()
        output = vmecpp.run(
            vmec_input,
            magnetic_field=self.response,
            restart_from=restart_from,
            verbose=False,
        )
        if self.verbose:
            print(f"[twin] converged in {time.monotonic() - started:.1f} s")

        if cache:
            with path.open("wb") as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output


    def solve_profiles(
        self,
        config: machine.Configuration | str,
        profiles,
        resolution: Resolution = SCAN,
        restart_from: vmecpp.VmecOutput | None = None,
        pressure_scale: float = 1.0,
        knots: int | None = None,
        cache: bool = True,
    ) -> vmecpp.VmecOutput:
        """Solve at the kinetic profiles' own pressure carried as a spline."""
        knots_s, knots_p = (
            profiles.pressure_spline() if knots is None else profiles.pressure_spline(knots)
        )
        return self.solve(
            self.state(
                config,
                scenario=Scenario.from_pressure_spline(knots_s, pressure_scale * knots_p),
            ),
            resolution,
            restart_from=restart_from,
            cache=cache,
        )

    def solve_input(
        self,
        vmec_input: vmecpp.VmecInput,
        key: str,
        restart_from: vmecpp.VmecOutput | None = None,
        cache: bool = True,
    ) -> vmecpp.VmecOutput:
        """Solve a prepared input, cached under an explicit key."""
        import vmecpp

        path = self.cache_dir / f"eq_{self._field_digest}_{key}.pkl"
        if cache and path.exists():
            if self.verbose:
                print(f"[twin] cache hit {path.name}")
            with path.open("rb") as handle:
                return pickle.load(handle)

        started = time.monotonic()
        output = vmecpp.run(
            vmec_input,
            magnetic_field=self.response,
            restart_from=restart_from,
            verbose=False,
        )
        if self.verbose:
            print(f"[twin] converged in {time.monotonic() - started:.1f} s")
        if cache:
            with path.open("wb") as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output


def _resize_boundary(
    vmec_input: vmecpp.VmecInput, mpol: int, ntor: int
) -> vmecpp.VmecInput:
    """Zero-pad or truncate the boundary and axis arrays to (mpol, ntor)."""

    def resize(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        out = np.zeros((mpol, 2 * ntor + 1))
        m_keep = min(mpol, arr.shape[0])
        n_old = (arr.shape[1] - 1) // 2
        n_keep = min(ntor, n_old)
        out[:m_keep, ntor - n_keep : ntor + n_keep + 1] = arr[
            :m_keep, n_old - n_keep : n_old + n_keep + 1
        ]
        return out

    def resize_axis(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        out = np.zeros(ntor + 1)
        keep = min(ntor + 1, arr.shape[0])
        out[:keep] = arr[:keep]
        return out

    for name in ("rbc", "zbs", "rbs", "zbc"):
        setattr(vmec_input, name, resize(getattr(vmec_input, name, None)))
    for name in ("raxis_c", "zaxis_s", "raxis_s", "zaxis_c"):
        setattr(vmec_input, name, resize_axis(getattr(vmec_input, name, None)))
    return vmec_input

# -- from boozer ------------------------------------------------------------------

MBOZ, NBOZ = 24, 16


@dataclasses.dataclass
class BoozerFile:
    """A written Boozer file and what it was produced from."""

    path: Path
    wout_path: Path
    num_surfaces: int
    mboz: int
    nboz: int
    #: True when the toroidal flux profile had to be filled in after writing.
    flux_repaired: bool


def _repair_toroidal_flux(boozmn: Path, wout: Path) -> bool:
    """Fill ``phi_b`` from the VMEC toroidal flux, and report whether it was needed."""
    import netCDF4

    with netCDF4.Dataset(wout, "r") as source:
        phi = np.asarray(source.variables["phi"][:], dtype=float)

    with netCDF4.Dataset(boozmn, "a") as target:
        if "phi_b" not in target.variables:
            return False
        stored = np.asarray(target.variables["phi_b"][:], dtype=float)
        if np.any(stored != 0.0):
            return False
        # booz_xform keeps the VMEC radial grid, so the profile transfers entry for
        # entry; a truncated surface list still starts at the axis.
        target.variables["phi_b"][: len(phi)] = phi[: len(stored)]
    return True


def write_boozer_file(
    output: vmecpp.VmecOutput,
    directory: str | Path,
    tag: str,
    mboz: int = MBOZ,
    nboz: int = NBOZ,
    verbose: bool = True,
) -> BoozerFile:
    """Transform every interior surface to Boozer coordinates and write ``boozmn.nc``."""
    import booz_xform

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wout_path = directory / f"wout_{tag}.nc"
    boozmn_path = directory / "boozmn.nc"
    output.wout.save(wout_path)

    transform = booz_xform.Booz_xform()
    transform.verbose = 0
    transform.read_wout(str(wout_path))
    transform.mboz = mboz
    transform.nboz = nboz
    # Surface 0 is the axis and carries no Boozer transformation.
    transform.compute_surfs = np.arange(1, int(output.wout.ns) - 1)
    transform.run()
    transform.write_boozmn(str(boozmn_path))

    repaired = _repair_toroidal_flux(boozmn_path, wout_path)
    if verbose:
        note = ", toroidal flux filled in" if repaired else ""
        print(
            f"[boozer] {boozmn_path} from {len(transform.compute_surfs)} surfaces "
            f"at mboz {mboz}, nboz {nboz}{note}"
        )
    return BoozerFile(
        path=boozmn_path,
        wout_path=wout_path,
        num_surfaces=len(transform.compute_surfs),
        mboz=mboz,
        nboz=nboz,
        flux_repaired=repaired,
    )
