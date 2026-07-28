"""Identified W7-X programmes and their published quantities, each entry carrying its source."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

# -- from programmes ---------------------------------------------------------------

#: In-vessel configuration each campaign ran in, resolved through machine.EPOCHS.
CAMPAIGN_OF_YEAR = {
    2015: "OP1.1",
    2016: "OP1.1",
    2017: "OP1.2a",
    2018: "OP1.2b",
    2022: "OP2.1",
    2023: "OP2.1",
    2024: "OP2.2",
    2025: "OP2.3",
}


@dataclasses.dataclass(frozen=True)
class Measured:
    """One published quantity, with the accuracy the source supports."""

    value: float
    unit: str
    #: Relative uncertainty. Values read from a figure or given with a tilde carry more
    #: than values stated to a digit, and the distinction is kept rather than averaged.
    relative_uncertainty: float
    source: str

    def band(self) -> tuple[float, float]:
        spread = abs(self.value) * self.relative_uncertainty
        return self.value - spread, self.value + spread


@dataclasses.dataclass(frozen=True)
class Programme:
    """One identified discharge."""

    identifier: str
    campaign: str
    configuration: str
    description: str
    source: str
    #: Quantities the source states in numbers, keyed by name.
    measured: dict[str, Measured] = dataclasses.field(default_factory=dict)
    #: Phase of the discharge the measured values belong to, in seconds.
    phase_s: tuple[float, float] | None = None

    @property
    def epoch(self) -> str:
        from w7x_twin.hardware import machine

        return machine.epoch_of_campaign(self.campaign).key


_OVERVIEW = (
    "T. Klinger et al., Overview of first Wendelstein 7-X high-performance operation, "
    "Nucl. Fusion 59, 112004 (2019), open copy at pure.mpg.de item_3156726"
)
_WOLF = (
    "R. C. Wolf et al., Performance of Wendelstein 7-X stellarator plasmas during the "
    "first divertor operation phase, Phys. Plasmas 26, 082504 (2019), open copy at "
    "pure.mpg.de item_3156726 component file_3158694"
)
_GLOBAL_GK = (
    "A. Banon Navarro et al., Global gyrokinetic analysis of a Wendelstein 7-X "
    "discharge, arXiv:2402.14403"
)
_ERROR_FIELD = (
    "S. A. Lazerson et al., Error fields in the Wendelstein 7-X stellarator, open copy "
    "at scipub.euro-fusion.org WPS1PR18_20965, table 1 and section 3.2"
)
_ISLAND_SOL = (
    "E. Maragkoudakis et al., On the interaction between the island divertor heat fluxes, "
    "the scrape-off layer radial electric field and the edge turbulence in Wendelstein 7-X "
    "plasmas, arXiv:2210.02134, section 2"
)
_LONG_PULSE = (
    "O. Grulke et al., Overview of the first Wendelstein 7-X long pulse campaign with "
    "fully water-cooled plasma facing components, Nucl. Fusion 64, 112002 (2024)"
)
_DIVERTOR_CONCEPT = (
    "H. Renner et al., Divertor concept for the Wendelstein 7-X stellarator: theoretical "
    "studies of the boundary and engineering, IAEA FT/P2-04"
)
_PELLET = (
    "S. A. Bozhenkov et al., High-performance plasmas after pellet injections in "
    "Wendelstein 7-X, Nucl. Fusion 60, 066011 (2020), open copy at pure.mpg.de "
    "item_3231111"
)
_SOL_TRANSPORT = (
    "D. Bold et al., Parametrisation of target heat flux distribution and study of "
    "transport parameters for boundary modelling in W7-X, arXiv:2201.06341"
)
_SYMMETRISATION = (
    "Compensation of 1/1 and 2/2 error field in Wendelstein 7-X via divertor heat load "
    "symmetrization, Nucl. Fusion, doi:10.1088/1741-4326/ae738e"
)

PROGRAMMES: tuple[Programme, ...] = (
    Programme(
        identifier="20180919.033",
        campaign="OP1.2b",
        configuration="standard",
        description=(
            "Two megawatts of electron-cyclotron heating replaced at 1.7 s by 3.4 MW of "
            "neutral-beam injection, which peaks the density strongly."
        ),
        source=_OVERVIEW,
        phase_s=(1.7, 3.5),
        measured={
            "stored_energy_ecrh_j": Measured(
                0.30e6, "J", 0.15, _WOLF + ", figure 11 discussion"
            ),
            "stored_energy_nbi_j": Measured(
                0.50e6, "J", 0.15, _WOLF + ", figure 11 discussion"
            ),
            "heating_power_ecrh_w": Measured(2.0e6, "W", 0.05, _OVERVIEW),
            "heating_power_nbi_w": Measured(3.4e6, "W", 0.05, _OVERVIEW),
            "central_density_late_m3": Measured(2.0e20, "m^-3", 0.15, _OVERVIEW),
        },
    ),
    Programme(
        identifier="20181010.036",
        campaign="OP1.2b",
        configuration="standard",
        description=(
            "Full high-power divertor detachment, triggered at 2.2 s by gas injection "
            "into the divertor region, with the other plasma quantities held."
        ),
        source=_OVERVIEW,
        phase_s=(2.2, 4.0),
        measured={
            "detachment_time_s": Measured(2.2, "s", 0.05, _OVERVIEW + ", figure 13"),
        },
    ),
    Programme(
        identifier="20181016.037",
        campaign="OP1.2b",
        configuration="standard",
        description=(
            "A gas-fuelled electron-cyclotron-heated discharge taken as representative of "
            "the standard scenario, analysed over the 4 to 5 s phase."
        ),
        source=_GLOBAL_GK,
        phase_s=(4.0, 5.0),
        measured={
            # The same programme also carries a pellet phase earlier in the discharge,
            # reported in the pellet paper. The two phases are different plasmas, so the
            # phase each number belongs to is named in its source rather than merged.
            "pellet_phase_stored_energy_j": Measured(
                1.15e6, "J", 0.05,
                _PELLET + ", figure 5, the diamagnetic energy after the first series, "
                "at 1.67 to 1.75 s rather than the 4 to 5 s phase analysed here",
            ),
            "pellet_phase_ion_temperature_ev": Measured(
                3.0e3, "eV", 0.10, _PELLET + ", figure 7a, the same earlier phase"
            ),
            "core_neoclassical_power_fraction": Measured(
                0.50, "", 0.20,
                _PELLET + ", about half the input power up to 30 cm in the post-pellet "
                "phase",
            ),
            "core_electron_neoclassical_power_fraction": Measured(
                0.30, "", 0.35,
                _PELLET + ", 20 to 40 per cent of the electron input power inside 30 cm",
            ),
        },
    ),
    Programme(
        identifier="20171207.006",
        campaign="OP1.2a",
        configuration="standard",
        description=(
            "Pellet series into a low-density electron-cyclotron-heated plasma, the "
            "power stepped to 4.9 MW at the end of the series. The post-pellet phase "
            "carries the enhanced confinement, and the highest triple product measured "
            "in W7-X."
        ),
        source=_PELLET,
        phase_s=(2.0, 2.5),
        measured={
            "heating_power_ecrh_w": Measured(4.9e6, "W", 0.05, _PELLET + ", section 3"),
            "stored_energy_ecrh_j": Measured(
                1.09e6, "J", 0.05, _PELLET + ", figure 2, the diamagnetic loop"
            ),
            "kinetic_energy_j": Measured(
                0.94e6, "J", 0.05,
                _PELLET + ", figure 2, the VMEC equilibrium against the same loop",
            ),
            "confinement_over_iss04": Measured(
                1.30, "", 0.10,
                _PELLET + ", about 30 per cent above the scaling on the plateau",
            ),
            "central_temperature_ev": Measured(
                3.0e3, "eV", 0.10,
                _PELLET + ", ion and electron temperatures equilibrated near 3 keV",
            ),
            "peak_density_m3": Measured(1.0e20, "m^-3", 0.10, _PELLET + ", section 3"),
            "volume_averaged_beta": Measured(0.01, "", 0.15, _PELLET + ", section 3"),
            "core_neoclassical_power_fraction": Measured(
                0.45, "", 0.15,
                _PELLET + ", 40 to 50 per cent of the input power in the core",
            ),
        },
    ),
    Programme(
        identifier="OP2.1 long pulse",
        campaign="OP2.1",
        configuration="standard",
        description=(
            "The eight-minute discharge that opened long-pulse operation on the "
            "water-cooled divertor: 3 MW of electron-cyclotron heating with the divertor "
            "attached, a shallow density ramp, and 1.3 GJ into the machine."
        ),
        source=_LONG_PULSE,
        measured={
            "heating_power_ecrh_w": Measured(3.0e6, "W", 0.05, _LONG_PULSE),
            "discharge_length_s": Measured(480.0, "s", 0.02, _LONG_PULSE),
            "heating_energy_j": Measured(1.3e9, "J", 0.05, _LONG_PULSE),
            "divertor_surface_temperature_c": Measured(
                650.0, "degC", 0.10,
                _LONG_PULSE + ", below the 1200 C the divertor is limited to",
            ),
        },
    ),
    Programme(
        identifier="OP2.1 detached long pulse",
        campaign="OP2.1",
        configuration="standard",
        description=(
            "Stationary detached operation for 110 s at 4 MW, held by neon puffs every "
            "two seconds, at a density of 10^20 m^-3."
        ),
        source=_LONG_PULSE,
        measured={
            "heating_power_ecrh_w": Measured(4.0e6, "W", 0.05, _LONG_PULSE),
            "discharge_length_s": Measured(110.0, "s", 0.02, _LONG_PULSE),
            "line_averaged_density_m3": Measured(1.0e20, "m^-3", 0.10, _LONG_PULSE),
        },
    ),
    Programme(
        identifier="20180911.033",
        campaign="OP1.2b",
        configuration="high_iota",
        description=(
            "Pellet series into an electron-cyclotron-heated plasma in the high-iota "
            "configuration, heated in O2 polarisation so the density could pass the X2 "
            "cut-off. The post-pellet enhancement matches the standard configuration's."
        ),
        source=_PELLET,
        measured={
            "stored_energy_ecrh_j": Measured(
                1.07e6, "J", 0.05, _PELLET + ", figure 5, the diamagnetic loop"
            ),
        },
    ),
    Programme(
        identifier="20180920.017",
        campaign="OP1.2b",
        configuration="standard",
        description=(
            "Gas-fuelled electron-cyclotron-heated discharge at the density and power of "
            "the pellet case, carried as its comparison. Its flat-top confinement is the "
            "gas-fuelled value, not the post-pellet one."
        ),
        source=_PELLET,
        phase_s=(2.5, 5.2),
        measured={
            "confinement_over_iss04": Measured(
                0.70, "", 0.10, _PELLET + ", figure 3b, typical of gas-fuelled operation"
            ),
        },
    ),
    Programme(
        identifier="20240918.036",
        campaign="OP2.2",
        configuration="standard",
        description=(
            "Compass scan over the trim coil phase in the forward field, taking the 1/1 "
            "correction to the setting that leaves the divertor load most even."
        ),
        source=_SYMMETRISATION,
        measured={
            "divertor_load_spread": Measured(
                0.27, "", 0.15,
                _SYMMETRISATION + ", the relative standard deviation of the load with the "
                "1/1 field corrected and the 2/2 field left alone",
            ),
            "divertor_load_spread_uncorrected": Measured(
                0.75, "", 0.15, _SYMMETRISATION + ", before either correction"
            ),
        },
    ),
    Programme(
        identifier="20240918.051",
        campaign="OP2.2",
        configuration="standard",
        description=(
            "The same forward-field scan carried on to the 2/2 harmonic, corrected with "
            "the in-vessel control coils rather than the trim coils."
        ),
        source=_SYMMETRISATION,
        measured={
            "divertor_load_spread": Measured(
                0.067, "", 0.15,
                _SYMMETRISATION + ", with both the 1/1 and the 2/2 field corrected",
            ),
        },
    ),
    Programme(
        identifier="20250218.064",
        campaign="OP2.3",
        configuration="standard",
        description=(
            "The same pair of corrections with the magnetic field reversed, where the same "
            "coil amplitudes leave nearly three times the residual load spread."
        ),
        source=_SYMMETRISATION,
        measured={
            "divertor_load_spread": Measured(
                0.18, "", 0.15,
                _SYMMETRISATION + ", reversed field, both corrections applied",
            ),
        },
    ),
    Programme(
        identifier="20180920.009",
        campaign="OP1.2b",
        configuration="standard",
        description=(
            "Low-density point of a three-discharge gas-fuelled density scan at 4.7 MW of "
            "electron-cyclotron heating, chosen for a low radiated fraction so the target "
            "load follows the cross-field transport rather than the radiation."
        ),
        source=_SOL_TRANSPORT,
        measured={
            "heating_power_ecrh_w": Measured(4.7e6, "W", 0.05, _SOL_TRANSPORT),
            "strike_line_width_m": Measured(
                0.03, "m", 0.33,
                _SOL_TRANSPORT + ", figure 12, the fitted narrow peak, 2 to 4 cm",
            ),
            "radiated_fraction": Measured(
                0.25, "", 0.40, _SOL_TRANSPORT + ", 0.15 to 0.35 across the scan"
            ),
            "divertor_power_w": Measured(
                3.5e6, "W", 0.15, _SOL_TRANSPORT + ", 3 to 4 MW measured by infrared"
            ),
            "toroidal_current_a": Measured(
                5.0e3, "A", 0.10, _SOL_TRANSPORT + ", reached after about six seconds"
            ),
        },
    ),
    Programme(
        identifier="20180920.013",
        campaign="OP1.2b",
        configuration="standard",
        description=(
            "Medium-density point of the same scan, whose strike-line width pattern "
            "follows the low-density one on both the upper and the lower targets."
        ),
        source=_SOL_TRANSPORT,
        measured={
            "heating_power_ecrh_w": Measured(4.7e6, "W", 0.05, _SOL_TRANSPORT),
            "strike_line_width_m": Measured(
                0.03, "m", 0.33,
                _SOL_TRANSPORT + ", figure 12, the fitted narrow peak, 2 to 4 cm",
            ),
            "toroidal_current_a": Measured(5.0e3, "A", 0.10, _SOL_TRANSPORT),
        },
    ),
    Programme(
        identifier="high-performance pellet phase",
        campaign="OP1.2b",
        configuration="standard",
        description=(
            "The pellet-fuelled discharge carrying the highest triple product observed in "
            "a stellarator, whose confinement is reported against the ISS04 scaling."
        ),
        source=_OVERVIEW,
        measured={
            "confinement_time_s": Measured(0.220, "s", 0.05, _OVERVIEW),
            "confinement_over_iss04": Measured(1.4, "", 0.07, _OVERVIEW),
            "triple_product_kev_m3_s": Measured(6.8e19, "keV m^-3 s", 0.1, _OVERVIEW),
            "central_beta": Measured(0.04, "", 0.25, _OVERVIEW),
        },
    ),
)

#: Machine-level quantities the same overview states, which are not tied to one discharge
#: but are checkable against what this package computes.
MACHINE_MEASUREMENTS: dict[str, Measured] = {
    # The island divertor puts a field line in one of two regimes depending on which side
    # of the island it sits, and the source states them separately rather than as one
    # number. A fan launched across the layer samples both, so both are carried.
    "island_connection_length_m": Measured(
        250.0, "m", 0.40,
        _ISLAND_SOL + ", a few hundred metres inside the island, between its O-point and "
        "the separatrix of the main plasma",
    ),
    "outer_island_connection_length_m": Measured(
        30.0, "m", 0.67,
        _ISLAND_SOL + ", of the order of tens of metres on the outer side of the island, "
        "where the lines reach the divertor plates",
    ),
    "divertor_peak_flux_rating_w_m2": Measured(
        10.0e6, "W/m^2", 0.0, _OVERVIEW + ", the water-cooled divertor design value"
    ),
    "energy_limit_test_divertor_j": Measured(
        80.0e6, "J", 0.0, _OVERVIEW + ", uncooled test divertor unit"
    ),
    # The design bound the target surfaces are shaped to. The geometry carries that
    # shaping, so this is what the measured incidence is checked against rather than what
    # it is set to.
    "divertor_incidence_degrees": Measured(
        3.0, "deg", 0.33, _DIVERTOR_CONCEPT + ", an incident angle of up to 3 degrees"
    ),
    # The width the power occupies at the target, fitted to infrared thermography, and the
    # cross-field transport a boundary code needs to reproduce it. The heat diffusivity is
    # three times the particle one throughout that study, so the 0.2 m^2/s it settles on is
    # 0.6 m^2/s of heat diffusivity.
    "strike_line_width_m": Measured(
        0.03, "m", 0.33,
        _SOL_TRANSPORT + ", figure 12, 2 to 4 cm over the divertor at 4.7 MW",
    ),
    "sol_perpendicular_diffusivity_m2_s": Measured(
        0.6, "m^2/s", 0.50,
        _SOL_TRANSPORT + ", chi = 3 D with D = 0.2 m^2/s, the value whose strike-line "
        "width matches the measurement most closely over a scan of 0.1 to 0.5",
    ),
    # The machine's intrinsic error field, as the normalised amplitude of the harmonic each
    # correction cancels. This is a resonant Fourier component of the field and not the
    # radial field on a midplane circle, so it is not the same normalisation as the n = 1
    # amplitude the errorfield analysis reports.
    "intrinsic_error_field_b11": Measured(
        0.5e-4, "", 0.30,
        _SYMMETRISATION + ", approximately 0.5e-4 for both b11 and b22",
    ),
    "divertor_load_spread_uncorrected": Measured(
        0.75, "", 0.15,
        _SYMMETRISATION + ", relative standard deviation of the divertor load with "
        "neither the 1/1 nor the 2/2 field corrected",
    ),
    "divertor_load_spread_corrected": Measured(
        0.067, "", 0.15,
        _SYMMETRISATION + ", the same measure with both corrections applied, which is the "
        "floor the compensation reaches",
    ),
    # The water-cooled divertor as it is actually run, which is the epoch this package's
    # default geometry carries.
    "divertor_measured_power_density_w_m2": Measured(
        6.0e6, "W/m^2", 0.15,
        _LONG_PULSE + ", the maximum load density demonstrated on the divertor modules",
    ),
    "divertor_steady_state_rating_w_m2": Measured(
        10.0e6, "W/m^2", 0.0, _LONG_PULSE + ", the steady-state specification"
    ),
    "strike_line_area_m2": Measured(
        1.0, "m^2", 0.20,
        _LONG_PULSE + ", the area the load has to spread over for 10 MW steady state",
    ),
    "divertor_local_power_density_w_m2": Measured(
        8.0e6, "W/m^2", 0.25,
        _DIVERTOR_CONCEPT + ", local power densities up to 8 MW/m^2 over a wide range of "
        "magnetic parameters, including low density and high temperature",
    ),
    "shafranov_shift_at_one_percent_beta_m": Measured(
        0.015, "m", 0.33,
        _OVERVIEW + ", 1 to 2 cm: the shift of the VMEC equilibrium computed at "
        "beta = 1 per cent with the standard theoretical pressure profile, whose "
        "peaking is 2, overlaid on an x-ray tomogram whose grid resolves 3 cm; the "
        "sentence credits the reduced Pfirsch-Schlueter current of the optimised field",
    ),
    # The machine's own equilibrium reconstruction, which is the only measured axis
    # displacement stated in numbers: order 1 cm for the bootstrap-current discharge at
    # its 346 kJ of kinetic energy, beta near 0.4 per cent. The overlay it is read from
    # shows the reconstructed surfaces against the vacuum ones, so the frames differ by
    # the boundary's own outward motion, inside the figure's own looseness.
    "reconstructed_axis_shift_m": Measured(
        0.010, "m", 0.5,
        "T. Andreeva et al., Equilibrium evaluation for Wendelstein 7-X experiment "
        "programs in the first divertor phase, Fusion Eng. Des. 146 (2019) 299, "
        "figure 5: the Minerva reconstruction of XP_20171108.040 at t = 9 s, "
        "W_kin = 346 kJ, shows a Shafranov shift of order 1 cm against the vacuum field",
    ),
    "effective_charge": Measured(
        1.5, "", 0.20, _PELLET + ", the measured value in the power balance sensitivity"
    ),
    "pellet_density_peaking": Measured(
        2.0, "1", 0.40,
        _PELLET + ", figure 12, the pellet discharges span n_e(0)/n_e(rho = 0.8) of 1.2 "
        "to 2.8 and the post-pellet phases sit at the top of it",
    ),
    "core_neoclassical_radius_m": Measured(
        0.30, "m", 0.10,
        _PELLET + ", the radius the core neoclassical fractions are quoted inside",
    ),
    "divertor_temperature_asymmetry_uncorrected": Measured(
        4.0, "1", 0.25, _ERROR_FIELD + ", a factor of almost four before correction"
    ),
    "divertor_temperature_asymmetry_corrected": Measured(
        2.0, "1", 0.25, _ERROR_FIELD + ", below two once the n = 1 field is applied"
    ),
}


#: Nonlinear gyrokinetic heat fluxes published for programme 20181016.037, as the power
#: through the flux surface. These are another code's results for the same discharge, not
#: measurements of it, and are kept apart from MACHINE_MEASUREMENTS for that reason: they
#: are what a second gyrokinetic treatment of this equilibrium returned, so they bound a
#: mixing-length estimate from a direction the machine's own numbers cannot.
GYROKINETIC_BENCHMARKS: dict[str, Measured] = {
    "flux_tube_ion_power_rho040_w": Measured(
        0.42e6, "W", 0.12, _GLOBAL_GK + ", table 2, nominal case"
    ),
    "flux_tube_electron_power_rho040_w": Measured(
        0.49e6, "W", 0.14, _GLOBAL_GK + ", table 2, nominal case"
    ),
    "flux_tube_ion_power_rho080_w": Measured(
        3.34e6, "W", 0.033, _GLOBAL_GK + ", table 1, flux tube"
    ),
    "flux_tube_electron_power_rho080_w": Measured(
        0.80e6, "W", 0.025, _GLOBAL_GK + ", table 1, flux tube"
    ),
    "global_ion_power_rho080_w": Measured(
        1.77e6, "W", 0.045, _GLOBAL_GK + ", table 1, radially global"
    ),
    "global_electron_power_rho080_w": Measured(
        0.30e6, "W", 0.067, _GLOBAL_GK + ", table 1, radially global"
    ),
    "electron_power_fraction_from_electron_gradient_rho040": Measured(
        0.85, "1", 0.06,
        _GLOBAL_GK + ", table 2, the electron channel falls about 85 per cent when the "
        "electron temperature gradient is removed",
    ),
}


@dataclasses.dataclass(frozen=True)
class TrimSetting:
    """A measured n = 1 trim waveform I_k = amplitude cos(2 pi (k - 1) / 5 - phase), with its planar current."""

    configuration: str
    planar_current_a: float
    amplitude_a: float
    phase_degrees: float
    method: str
    source: str = _ERROR_FIELD


#: Trim coil waveforms measured to give the most symmetric divertor load, and the one
#: measured on the magnetic topology rather than the heat load. These are corrections: the
#: machine's own n = 1 error field is what they cancel, so applying the negative of one to
#: an ideal coil set synthesises the error field it was measured against.
TRIM_SETTINGS: tuple[TrimSetting, ...] = (
    TrimSetting("standard", -700.0, 98.0, 162.0, "divertor thermocouple compass scan"),
    TrimSetting(
        "narrow_mirror", -750.0, 99.0, 163.0, "divertor thermocouple compass scan"
    ),
    TrimSetting("high_iota", -9790.0, 134.0, 180.0, "divertor thermocouple compass scan"),
    TrimSetting(
        "high_iota", -9790.0, 123.0, 174.0,
        "flux surface mapping compass scan on the axis displacement",
    ),
)


@dataclasses.dataclass(frozen=True)
class SymmetrisationSetting:
    """A load-spread-minimising correction; ``mode_phase_degrees`` is the harmonic's phase,
    not the coil waveform's."""

    campaign: str
    field_sense: str
    mode: str
    circuit: str
    coil_current_a: float
    mode_phase_degrees: float
    intrinsic_phase_degrees: float
    spread_before: float
    spread_after: float
    programme: str
    source: str = _SYMMETRISATION


#: Corrections measured on the divertor load in the actively cooled epoch, for both the
#: 1/1 harmonic the trim coils reach and the 2/2 harmonic the in-vessel control coils do.
#: The reversed-field entries are the same machine run with the field reversed, where the
#: same amplitudes leave three times the residual spread.
SYMMETRISATION_SETTINGS: tuple[SymmetrisationSetting, ...] = (
    SymmetrisationSetting(
        campaign="OP2.2", field_sense="forward", mode="1/1", circuit="trim",
        coil_current_a=-109.0, mode_phase_degrees=-18.0,
        intrinsic_phase_degrees=162.0, spread_before=0.75, spread_after=0.27,
        programme="20240918.036",
    ),
    SymmetrisationSetting(
        campaign="OP2.2", field_sense="forward", mode="2/2", circuit="control",
        coil_current_a=480.0, mode_phase_degrees=54.0,
        intrinsic_phase_degrees=-126.0, spread_before=0.27, spread_after=0.067,
        programme="20240918.051",
    ),
    SymmetrisationSetting(
        campaign="OP2.3", field_sense="reversed", mode="1/1", circuit="trim",
        coil_current_a=-109.0, mode_phase_degrees=126.0,
        intrinsic_phase_degrees=float("nan"), spread_before=0.48, spread_after=0.45,
        programme="20250218.056",
    ),
    SymmetrisationSetting(
        campaign="OP2.3", field_sense="reversed", mode="2/2", circuit="control",
        coil_current_a=480.0, mode_phase_degrees=-162.0,
        intrinsic_phase_degrees=18.0, spread_before=0.45, spread_after=0.18,
        programme="20250218.064",
    ),
)


def symmetrisation_settings(
    field_sense: str | None = None, mode: str | None = None
) -> tuple[SymmetrisationSetting, ...]:
    """The measured load-symmetrisation corrections, filtered by field sense and mode."""
    return tuple(
        setting
        for setting in SYMMETRISATION_SETTINGS
        if (field_sense is None or setting.field_sense == field_sense)
        and (mode is None or setting.mode == mode)
    )


def trim_setting(configuration: str, method: str | None = None) -> TrimSetting:
    """The measured trim waveform for a configuration, optionally by method."""
    for setting in TRIM_SETTINGS:
        if setting.configuration == configuration and (
            method is None or method == setting.method
        ):
            return setting
    raise KeyError(
        f"no trim setting for {configuration!r}"
        + (f" by {method!r}" if method else "")
        + f"; have {sorted({s.configuration for s in TRIM_SETTINGS})}"
    )


def get(identifier: str) -> Programme:
    for programme in PROGRAMMES:
        if programme.identifier == identifier:
            return programme
    raise KeyError(
        f"no programme {identifier!r}; have {[p.identifier for p in PROGRAMMES]}"
    )


def identifiers() -> list[str]:
    return [programme.identifier for programme in PROGRAMMES]

# -- from profiles -----------------------------------------------------------------

# Measured profiles digitised from the published figures, with the bands they carry.
#
# Most of what the publications state about a discharge is a number in the text, and
# :mod:`w7x_twin.records.programmes` holds those. The profiles are not: they are drawn. A
# figure in a vector document carries its curves as coordinates, so they can be read off
# exactly rather than sampled by eye, and the shaded band the authors draw around a profile
# comes with them as the uncertainty they assign it.
#
# ``tools/digitise_profiles.py`` produced ``thomson_profiles.json`` beside this module, and
# the residual of each axis against a straight line through its own tick labels is recorded
# with the curves, since one of the abscissae is a laser path and is not linear in the flux
# coordinate the ticks are labelled with.

PROFILES = Path(__file__).with_name("thomson_profiles.json")


@dataclasses.dataclass(frozen=True)
class MeasuredProfile:
    """One digitised curve, and the band the figure draws around it."""

    discharge: str
    quantity: str
    unit: str
    label: str
    abscissa: str
    x: np.ndarray
    y: np.ndarray
    band_x: np.ndarray | None
    band_low: np.ndarray | None
    band_high: np.ndarray | None
    figure: str
    source: str
    #: Largest departure of the axis tick labels from a straight line through them, in the
    #: units of that axis. Large where the abscissa is not linear in what it is labelled by.
    axis_residual: dict[str, float] = dataclasses.field(default_factory=dict)
    phase_s: tuple[float, float] | None = None

    def at(self, x) -> np.ndarray:
        """The curve at given abscissa values, outside its range as its end value."""
        return np.interp(np.asarray(x, dtype=float), self.x, self.y)

    def uncertainty_at(self, x) -> np.ndarray:
        """Half-width of the drawn band at given abscissa values, in the curve's unit."""
        if self.band_x is None:
            return np.zeros_like(np.asarray(x, dtype=float))
        x = np.asarray(x, dtype=float)
        return 0.5 * (
            np.interp(x, self.band_x, self.band_high)
            - np.interp(x, self.band_x, self.band_low)
        )

    def peaking(self, edge: float = 0.8) -> float:
        """Centre value over the value at one abscissa fraction, as W7-X reports it."""
        span = float(np.max(np.abs(self.x)))
        centre = float(self.at(0.0))
        return centre / float(self.at(edge * span))


def load(path: str | Path = PROFILES) -> list[MeasuredProfile]:
    """Every digitised profile, one entry per curve."""
    stored = json.loads(Path(path).read_text())
    out: list[MeasuredProfile] = []
    for figure in stored["figures"]:
        name = f"{figure['figure']}{figure.get('panel', '')}"
        for series in figure["series"]:
            band_x = series.get("band_x")
            out.append(
                MeasuredProfile(
                    discharge=figure["discharge"],
                    quantity=figure["quantity"],
                    unit=figure["unit"],
                    label=series["label"],
                    abscissa=figure["abscissa"],
                    x=np.asarray(series["x"], dtype=float),
                    y=np.asarray(series["y"], dtype=float),
                    band_x=None if band_x is None else np.asarray(band_x, dtype=float),
                    band_low=None
                    if band_x is None
                    else np.asarray(series["band_low"], dtype=float),
                    band_high=None
                    if band_x is None
                    else np.asarray(series["band_high"], dtype=float),
                    figure=name,
                    source=stored["source"],
                    axis_residual=figure.get("axis_residual", {}),
                    phase_s=tuple(figure.get("phase_s") or series.get("phase_s") or ())
                    or None,
                )
            )
    return out


def find(discharge: str, quantity: str, label: str | None = None) -> MeasuredProfile:
    """One digitised curve by discharge and quantity."""
    for profile in load():
        if profile.discharge != discharge or profile.quantity != quantity:
            continue
        if label is None or label.lower() in profile.label.lower():
            return profile
    raise KeyError(f"no digitised {quantity} for {discharge}" + (f" ({label})" if label else ""))
