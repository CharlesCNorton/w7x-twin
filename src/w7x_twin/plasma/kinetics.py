"""Kinetic profiles, their pressure, and the carbon impurity content and radiation."""

from __future__ import annotations

import dataclasses

import numpy as np

ELEMENTARY_CHARGE = 1.602176634e-19


# -- from impurities ---------------------------------------------------------------

# Mavrin polynomial fits [Plasma Phys. Rep. 43 (2017) 1023], coefficients from radas (CFS, MIT licence); P = n_e n_Z L_Z.


#: Nuclear charge of the impurity these fits describe.
CARBON_Z = 6


def log_gradient(values: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """d ln(values)/d(radius), the argument floored away from zero."""
    return np.gradient(np.log(np.maximum(values, 1e-30)), radius)


@dataclasses.dataclass(frozen=True)
class MavrinFit:
    """Mavrin polynomial 10^F(X, Y) with X = log10 T_e[eV], Y = log10(n_e tau / 1e19) capped at 0."""

    t_min_ev: tuple[float, ...]
    t_max_ev: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]

    def __call__(
        self, temperature_ev: np.ndarray, ne_tau: float = 1.0e19
    ) -> np.ndarray:
        temperature = np.clip(
            np.asarray(temperature_ev, dtype=float), self.t_min_ev[0], self.t_max_ev[-1]
        )
        x = np.log10(temperature)
        y = np.minimum(np.log10(max(ne_tau, 1.0e15) / 1.0e19), 0.0)

        # Abutting intervals: either edge assignment gives the same value.
        edges = np.array(self.t_max_ev[:-1])
        interval = np.searchsorted(edges, temperature, side="right")
        interval = np.clip(interval, 0, len(self.t_max_ev) - 1)

        a = np.array(self.coefficients)[:, interval]
        exponent = (
            a[0]
            + a[1] * x
            + a[2] * y
            + a[3] * x**2
            + a[4] * x * y
            + a[5] * y**2
            + a[6] * x**3
            + a[7] * x**2 * y
            + a[8] * x * y**2
            + a[9] * y**3
        )
        return np.power(10.0, exponent)


#: Radiative cooling rate of carbon, in W m^3.
CARBON_COOLING_RATE = MavrinFit(
    t_min_ev=(1, 7, 20, 70, 200, 700),
    t_max_ev=(7, 20, 70, 200, 700, 15000),
    coefficients=(
        (-3.4509e01, -4.9228e01, -1.9100e01, -6.7743e01, -2.4016e01, -2.8126e01),
        (6.7599e00, 5.3922e01, -1.5476e01, 4.1606e01, -7.3974e00, -4.1679e00),
        (-1.7140e-02, 8.4584e-01, 4.2962e00, -5.3665e00, 2.9707e00, 4.9937e-01),
        (-4.0337e00, -5.1128e01, 2.1893e00, -1.5734e01, 1.6859e00, 9.0578e-01),
        (1.5517e-01, -8.9366e-01, -6.1658e00, 6.1760e00, -2.1965e00, -5.3687e-01),
        (2.1110e-02, -2.2710e-02, 1.6098e-01, 7.8010e-01, 3.0521e-01, 2.5962e-01),
        (6.5977e-01, 1.4758e01, 1.1021e00, 1.7905e00, -1.1147e-01, -5.8310e-02),
        (-1.7392e-01, 1.6371e-01, 2.1568e00, -1.7320e00, 3.8653e-01, 1.0420e-01),
        (-2.9270e-02, 2.9362e-01, 1.1101e-01, -2.7897e-01, 3.8970e-02, 4.6610e-02),
        (1.7600e-03, 5.5880e-02, 4.2700e-02, 2.3450e-02, 7.8690e-02, 7.3950e-02),
    ),
)

#: Mean charge of carbon in the same balance, between 0 and 6.
CARBON_MEAN_CHARGE = MavrinFit(
    t_min_ev=(1, 3, 10, 30, 100, 300),
    t_max_ev=(3, 10, 30, 100, 300, 15000),
    coefficients=(
        (-1.7799e-01, 6.8333e-01, -1.3092e00, 1.7808e00, -4.7139e00, 4.1877e-01),
        (2.4465e00, -2.0893e00, 4.1883e00, -1.3260e00, 6.0788e00, 3.4803e-01),
        (1.8370e-02, 1.4225e-01, 1.1598e-01, 5.1150e-01, -5.7445e-01, 1.3581e-01),
        (-1.2305e01, 3.0963e00, -3.0465e00, 2.3885e-01, -2.1878e00, -1.1064e-01),
        (-1.2788e00, -4.8037e-01, -1.8574e-01, -6.8025e-01, 6.2149e-01, -8.0230e-02),
        (-7.4430e-02, -3.0240e-02, -3.5430e-02, -2.3040e-02, 5.3540e-02, 3.0200e-03),
        (1.9589e01, -1.1135e00, 7.3542e-01, 8.0050e-02, 2.5342e-01, 1.1530e-02),
        (3.3268e00, 3.5170e-01, 7.5470e-02, 2.1915e-01, -1.6397e-01, 1.2450e-02),
        (3.2173e-01, 2.1640e-02, 3.2500e-02, 6.5800e-03, -3.4550e-02, 3.8400e-03),
        (1.7020e-02, 2.2200e-03, 4.7400e-03, -1.6500e-03, -1.8900e-03, 6.9500e-03),
    ),
)


def mean_charge(temperature_ev: np.ndarray, ne_tau: float = 1.0e19) -> np.ndarray:
    """Mean carbon charge, clipped to the physical range the fit is bounded by."""
    return np.clip(CARBON_MEAN_CHARGE(temperature_ev, ne_tau), 0.0, float(CARBON_Z))


def cooling_rate(temperature_ev: np.ndarray, ne_tau: float = 1.0e19) -> np.ndarray:
    """Carbon radiative cooling rate in W m^3, so P_rad = n_e n_C L_Z."""
    return CARBON_COOLING_RATE(temperature_ev, ne_tau)


@dataclasses.dataclass
class Composition:
    """Densities and charges of a hydrogen plasma carrying carbon, main ions fixed by quasineutrality."""

    electron_density_m3: np.ndarray
    impurity_density_m3: np.ndarray
    ion_density_m3: np.ndarray
    charge: np.ndarray
    z_effective: np.ndarray

    @property
    def dilution(self) -> np.ndarray:
        """n_i / n_e, which is one when there is no impurity."""
        return self.ion_density_m3 / np.maximum(self.electron_density_m3, 1e-30)


def composition(
    electron_density_m3: np.ndarray,
    electron_temperature_ev: np.ndarray,
    fraction: float,
    ne_tau: float = 1.0e19,
) -> Composition:
    """Charge state, dilution and effective charge of a carbon-seeded hydrogen plasma."""
    n_e = np.asarray(electron_density_m3, dtype=float)
    charge = mean_charge(electron_temperature_ev, ne_tau)
    n_impurity = fraction * n_e
    n_ion = np.maximum(n_e - charge * n_impurity, 0.0)
    z_effective = (n_ion + n_impurity * charge**2) / np.maximum(n_e, 1e-30)
    return Composition(
        electron_density_m3=n_e,
        impurity_density_m3=n_impurity,
        ion_density_m3=n_ion,
        charge=charge,
        z_effective=z_effective,
    )


#: NRL formulary bremsstrahlung coefficient P = 1.69e-32 n_e sqrt(T_e) sum Z^2 n_Z, converted to SI.
BREMSSTRAHLUNG_COEFFICIENT = 1.69e-38


def bremsstrahlung(
    electron_density_m3: np.ndarray,
    electron_temperature_ev: np.ndarray,
    ion_density_m3: np.ndarray,
    ion_charge: float = 1.0,
) -> np.ndarray:
    """Main-ion bremsstrahlung in W/m^3; the impurity's own is inside its cooling rate."""
    return (
        BREMSSTRAHLUNG_COEFFICIENT
        * np.asarray(electron_density_m3, dtype=float)
        * np.sqrt(np.maximum(np.asarray(electron_temperature_ev, dtype=float), 0.0))
        * ion_charge**2
        * np.asarray(ion_density_m3, dtype=float)
    )


def radiated_power_density(
    parts: Composition,
    electron_temperature_ev: np.ndarray,
    ne_tau: float = 1.0e19,
) -> dict[str, np.ndarray]:
    """Radiated power per unit volume, split by channel, in W/m^3."""
    line = (
        parts.electron_density_m3
        * parts.impurity_density_m3
        * cooling_rate(electron_temperature_ev, ne_tau)
    )
    brems = bremsstrahlung(
        parts.electron_density_m3, electron_temperature_ev, parts.ion_density_m3
    )
    return {"impurity": line, "bremsstrahlung": brems, "total": line + brems}


def carbon_profile(
    s: np.ndarray,
    electron_density_m3: np.ndarray,
    minor_radius_m: float,
    edge_fraction: float,
    diffusivity_m2_s: float = 0.5,
    pinch_m_s: float = -0.3,
    floor: float = 1.0e-4,
) -> np.ndarray:
    """Carbon fraction per surface from a target-edge source: zero-flux solution of Gamma = -D dn/dr + v n."""
    s = np.asarray(s, dtype=float)
    n_e = np.asarray(electron_density_m3, dtype=float)
    radius = minor_radius_m * np.sqrt(np.clip(s, 0.0, 1.0))

    # Zero core flux: d ln n_C / dr = v / D, integrated inward from the edge.
    ratio = pinch_m_s / max(diffusivity_m2_s, 1e-12)
    density = np.empty_like(radius)
    density[-1] = edge_fraction * n_e[-1]
    for index in range(len(radius) - 1, 0, -1):
        step = radius[index] - radius[index - 1]
        density[index - 1] = density[index] * np.exp(-ratio * step)

    return np.maximum(density / np.maximum(n_e, 1e-30), floor)


# -- from kinetics -----------------------------------------------------------------


@dataclasses.dataclass
class KineticProfiles:
    """Electron density and both temperatures against normalised toroidal flux, in m^-3 and eV."""

    density_axis_m3: float = 8.0e19
    density_edge_m3: float = 0.5e19
    density_flatness: float = 5.0
    density_exponent: float = 0.5
    #: Measured density knots as (s, m^-3) pairs; when present they are the profile.
    density_points: tuple[tuple[float, float], ...] | None = None

    electron_temperature_axis_ev: float = 3500.0
    ion_temperature_axis_ev: float = 1800.0
    temperature_edge_ev: float = 100.0
    temperature_exponent: float = 2.0

    #: n_C / n_e; zero is a pure hydrogen plasma.
    carbon_fraction: float = 0.0
    #: True solves the carbon profile inward from the edge value; False holds it flat.
    carbon_from_target: bool = False
    #: Second main ion species as a share of the fuel ions, with its mass in amu.
    second_ion_fraction: float = 0.0
    second_ion_mass_amu: float = 2.0
    #: Minor radius the carbon transport is solved over, in metres.
    minor_radius_m: float = 0.49

    def carbon_profile(self, s: np.ndarray) -> np.ndarray:
        """Carbon fraction at each surface, flat or carried in from the targets."""
        s = np.asarray(s, dtype=float)
        if self.carbon_fraction == 0.0:
            return np.zeros_like(s)
        if not self.carbon_from_target:
            return np.full_like(s, self.carbon_fraction)
        return carbon_profile(
            s, self.density(s), self.minor_radius_m, self.carbon_fraction
        )

    def mean_ion_mass_amu(self, s: np.ndarray) -> np.ndarray:
        """Fuel-averaged ion mass, which the sound speed and gyro-Bohm scale need."""
        s = np.asarray(s, dtype=float)
        return np.full_like(
            s, 1.0 + self.second_ion_fraction * (self.second_ion_mass_amu - 1.0)
        )

    def composition(self, s: np.ndarray):
        """Charge state, dilution and effective charge at each surface."""
        return composition(
            self.density(s), self.electron_temperature(s), self.carbon_profile(s)
        )

    @property
    def z_effective(self) -> float:
        """Effective charge at mid-radius, the scalar the analytic bootstrap formula takes."""
        if self.carbon_fraction == 0.0:
            return 1.0
        return float(self.composition(np.array([0.5])).z_effective[0])

    def peaking(self, rho: float = 0.8) -> float:
        """Density peaking, ``n_e(0) / n_e(rho)``, the ratio W7-X reports its profiles by."""
        return float(self.density(np.array([0.0]))[0] / self.density(np.array([rho**2]))[0])

    def with_peaking(
        self, factor: float, rho: float = 0.8, iterations: int = 80
    ) -> "KineticProfiles":
        """Same profiles with the density flatness bisected to carry a given peaking."""
        low, high = 0.05, 60.0
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            trial = dataclasses.replace(self, density_flatness=middle)
            if trial.peaking(rho) > factor:
                low = middle
            else:
                high = middle
        return dataclasses.replace(self, density_flatness=0.5 * (low + high))

    def z_effective_profile(self, s: np.ndarray) -> np.ndarray:
        if self.carbon_fraction == 0.0:
            return np.ones_like(np.asarray(s, dtype=float))
        return self.composition(s).z_effective

    def density(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        if self.density_points is not None:
            knots = np.asarray(self.density_points, dtype=float)
            return np.interp(s, knots[:, 0], knots[:, 1])
        shape = np.maximum(1.0 - s**self.density_flatness, 0.0) ** self.density_exponent
        return self.density_edge_m3 + (
            self.density_axis_m3 - self.density_edge_m3
        ) * shape

    def electron_temperature(self, s: np.ndarray) -> np.ndarray:
        return self._temperature(s, self.electron_temperature_axis_ev)

    def ion_temperature(self, s: np.ndarray) -> np.ndarray:
        return self._temperature(s, self.ion_temperature_axis_ev)

    def _temperature(self, s: np.ndarray, axis_value: float) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        shape = (1.0 - s) ** self.temperature_exponent
        return self.temperature_edge_ev + (
            axis_value - self.temperature_edge_ev
        ) * shape

    def pressure_pa(self, s: np.ndarray) -> np.ndarray:
        """Total pressure of electrons, diluted main ions and the impurity, in Pa."""
        electron = self.electron_temperature(s)
        ion = self.ion_temperature(s)
        if self.carbon_fraction == 0.0:
            density = self.density(s)
            return ELEMENTARY_CHARGE * density * (electron + ion)
        parts = self.composition(s)
        return ELEMENTARY_CHARGE * (
            parts.electron_density_m3 * electron
            + (parts.ion_density_m3 + parts.impurity_density_m3) * ion
        )

    def scaled(self, factor: float) -> KineticProfiles:
        """Same profile shapes with both temperatures scaled, so beta scales with them."""
        return dataclasses.replace(
            self,
            electron_temperature_axis_ev=self.electron_temperature_axis_ev * factor,
            ion_temperature_axis_ev=self.ion_temperature_axis_ev * factor,
        )

    def as_simsopt(self, num_knots: int = 51):
        """(ne, Te, Ti) as simsopt profile objects for the bootstrap formula."""
        from simsopt.mhd.profiles import ProfileSpline

        s = np.linspace(0.0, 1.0, num_knots)
        return (
            ProfileSpline(s, self.density(s)),
            ProfileSpline(s, self.electron_temperature(s)),
            ProfileSpline(s, self.ion_temperature(s)),
        )

    def z_effective_as_simsopt(self, num_knots: int = 51):
        """The effective charge as a profile the bootstrap formula can take."""
        from simsopt.mhd.profiles import ProfileSpline

        s = np.linspace(0.0, 1.0, num_knots)
        return ProfileSpline(s, self.z_effective_profile(s))

    def pressure_spline(self, num_knots: int = 41) -> tuple[np.ndarray, np.ndarray]:
        """Knots for VMEC's spline pressure profile, in s and pascals."""
        s = np.linspace(0.0, 1.0, num_knots)
        return s, self.pressure_pa(s)


#: Representative of W7-X high-performance operation.
HIGH_PERFORMANCE = KineticProfiles()

#: A lower-power case: same shapes, temperatures halved.
MODERATE = HIGH_PERFORMANCE.scaled(0.5)
