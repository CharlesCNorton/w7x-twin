"""Guards on the conventions that reverse silently."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from w7x_twin.hardware import machine
from w7x_twin.magnetics import field
from w7x_twin.plasma import current, kinetics, neoclassical
from w7x_twin.hardware.walls import base_name, unit_of
from w7x_twin.hardware.walls import Vessel


# -- full-torus rewriting ---------------------------------------------------

def test_expand_2d_moves_each_mode_to_n_times_nfp():
    """The rewriting is exact: mode n becomes n*nfp and nothing else changes."""
    mpol, ntor, nfp = 4, 3, 5
    coeff = np.zeros((mpol, 2 * ntor + 1))
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            coeff[m, ntor + n] = 100 * m + n

    wide = field._expand_2d(coeff, nfp, ntor * nfp)
    assert wide.shape == (mpol, 2 * ntor * nfp + 1)
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            assert wide[m, ntor * nfp + n * nfp] == coeff[m, ntor + n]
    # Everything the rewriting did not place is zero, and nothing is lost.
    assert np.count_nonzero(wide) == np.count_nonzero(coeff)
    assert np.sum(wide) == pytest.approx(np.sum(coeff))


def test_expand_axis_places_the_same_modes():
    nfp, ntor = 5, 3
    axis = np.array([1.0, 2.0, 3.0, 4.0])
    wide = field._expand_axis(axis, nfp, ntor * nfp)
    assert wide[0] == 1.0
    assert wide[nfp] == 2.0
    assert wide[2 * nfp] == 3.0
    assert np.count_nonzero(wide) == 4


# -- Chandrasekhar function -------------------------------------------------

def test_chandrasekhar_limits():
    """G(x) -> 2x/(3 sqrt(pi)) as x -> 0 and -> 1/(2x^2) as x -> infinity."""
    small = np.array([1e-4, 1e-3, 1e-2])
    assert neoclassical._chandrasekhar(small) == pytest.approx(
        2.0 * small / (3.0 * np.sqrt(np.pi)), rel=1e-3
    )
    large = np.array([6.0, 10.0])
    assert neoclassical._chandrasekhar(large) == pytest.approx(
        1.0 / (2.0 * large**2), rel=1e-3
    )


def test_chandrasekhar_is_finite_at_zero():
    assert neoclassical._chandrasekhar(np.array([0.0]))[0] == 0.0


def test_chandrasekhar_peaks_once():
    x = np.linspace(1e-3, 6.0, 400)
    g = neoclassical._chandrasekhar(x)
    assert np.count_nonzero(np.diff(np.sign(np.diff(g))) < 0) == 1


# -- monoenergetic interpolation --------------------------------------------

def _table(d31_values, nu):
    return neoclassical.MonoenergeticCoefficients(
        s=0.2,
        collisionality=np.asarray(nu),
        radial_field=np.zeros_like(nu, dtype=float),
        d11=np.full_like(np.asarray(nu, dtype=float), 1.0),
        d31=np.asarray(d31_values, dtype=float),
        d33=np.zeros_like(nu, dtype=float),
        d33_spitzer=np.zeros_like(nu, dtype=float),
    )


def test_d31_interpolation_passes_through_zero():
    """D_31 changes sign inside the table, so the interpolant must reach zero."""
    nu = np.array([1e-4, 1e-3])
    table = _table([0.1, -0.1], nu)
    probe = np.geomspace(1e-4, 1e-3, 41)
    values = neoclassical._interpolate_coefficient(
        table, probe, np.zeros_like(probe), "d31"
    )
    assert values[0] == pytest.approx(0.1)
    assert values[-1] == pytest.approx(-0.1)
    assert np.min(np.abs(values)) < 5e-3


def test_d11_continues_as_the_measured_power_law():
    """Below the table D_11 follows the exponent its end points measure."""
    nu = np.array([1e-5, 1e-4, 1e-3])
    table = neoclassical.MonoenergeticCoefficients(
        s=0.2,
        collisionality=nu,
        radial_field=np.zeros(3),
        d11=np.array([1e-1, 1e-2, 1e-3]),  # exactly 1/nu
        d31=np.zeros(3),
        d33=np.zeros(3),
        d33_spitzer=np.zeros(3),
    )
    # D_11 nu is constant at 1e-6, so the continuation gives 1e-6 / nu below the table.
    probe = np.array([1e-7, 1e-6])
    values = neoclassical._interpolate_coefficient(
        table, probe, np.zeros_like(probe), "d11"
    )
    assert values[0] == pytest.approx(10.0, rel=1e-6)
    assert values[1] == pytest.approx(1.0, rel=1e-6)
    assert values[0] * probe[0] == pytest.approx(values[1] * probe[1], rel=1e-9)


def test_field_axis_interpolates_between_slices():
    """The electric-field axis is used, not snapped to its nearest tabulated value."""
    nu = np.array([1e-4, 1e-4])
    table = neoclassical.MonoenergeticCoefficients(
        s=0.2,
        collisionality=nu,
        radial_field=np.array([0.0, 1e-3]),
        d11=np.array([1.0, 0.1]),
        d31=np.zeros(2),
        d33=np.zeros(2),
        d33_spitzer=np.zeros(2),
    )
    probe = np.array([1e-4])
    middle = neoclassical._interpolate_coefficient(
        table, probe, np.array([5e-4]), "d11"
    )[0]
    assert 0.1 < middle < 1.0


# -- winding numbers --------------------------------------------------------

def test_every_circuit_declares_its_winding_number():
    declared = {c.key: c.turns for c in machine.MAIN_CIRCUITS + machine.AUXILIARY_CIRCUITS}
    assert declared["npc1"] == 108
    assert declared["pca"] == 36
    # The type B trim coil carries 72 turns, not the 48 of the type A coils.
    assert declared["trim_a1"] == 48
    assert declared["trim_b1"] == 72
    assert declared["cc1u"] == 8


def test_declared_turns_match_the_constructed_geometry():
    """The constructed ampere-turns equal the winding number the circuit declares."""
    from w7x_twin.hardware import coils as aux

    declared = {c.key: c.turns for c in machine.AUXILIARY_CIRCUITS}
    for group in aux.trim_coils():
        assert group.turns * len(group.filaments) == declared[group.key]


def test_ampere_turns_are_the_product():
    circuit = machine.MAIN_CIRCUITS[0]
    current_per_turn = 13000.0
    assert circuit.turns * current_per_turn == pytest.approx(1.404e6)


# -- vessel containment -----------------------------------------------------

def _square_vessel():
    """A square annulus cross-section, so containment has an analytic answer."""
    contour_r = np.array([5.0, 6.0, 6.0, 5.0])
    contour_z = np.array([-0.5, -0.5, 0.5, 0.5])
    return Vessel(
        phi=np.array([0.0, 2 * np.pi / 5]),
        r=np.vstack([contour_r, contour_r]),
        z=np.vstack([contour_z, contour_z]),
        num_field_periods=5,
    )


def test_containment_separates_inside_from_outside():
    resampled = _square_vessel().resample(np.array([0.0]))
    inside = np.array([5.5, 5.1, 5.9])
    assert not resampled.outside(inside, np.zeros(3), 0).any()
    outside = np.array([4.9, 6.1, 5.5])
    z = np.array([0.0, 0.0, 0.6])
    assert resampled.outside(outside, z, 0).all()


def test_non_finite_points_count_as_outside():
    resampled = _square_vessel().resample(np.array([0.0]))
    assert resampled.outside(np.array([np.nan]), np.array([0.0]), 0)[0]
    assert resampled.outside(np.array([5.5]), np.array([np.nan]), 0)[0]


# -- strike attribution -----------------------------------------------------

def test_unit_of_maps_angle_to_module_and_side():
    period = 2 * np.pi / 5
    phi = np.array([0.1, period + 0.1, 2 * period + 0.1, 4 * period + 0.1])
    z = np.array([0.5, -0.5, 0.5, -0.5])
    module, upper = unit_of(phi, z)
    assert module.tolist() == [1, 2, 3, 5]
    assert upper.tolist() == [True, False, True, False]


def test_unit_of_reports_module_zero_for_a_line_that_never_struck():
    module, _ = unit_of(np.array([np.nan]), np.array([np.nan]))
    assert module[0] == 0


def test_base_name_drops_only_the_unit_qualifier():
    assert base_name("divertor horizontal target, upper") == "divertor horizontal target"
    assert base_name("divertor horizontal target, lower") == "divertor horizontal target"
    assert base_name("baffle, horizontal mid") == "baffle, horizontal mid"


# -- machine epochs ---------------------------------------------------------

def test_epoch_versions_are_distinct():
    """Two epochs over the same files produce different geometry versions."""

    common = {
        "coils_path": "pyproject.toml",  # any file; only its digest is used
        "grid_parameters": {"num_r": 121},
    }
    tdu = machine.geometry_version(**common, epoch_key="tdu")
    hhf = machine.geometry_version(**common, epoch_key="hhf")
    assert tdu.digest != hhf.digest
    # The coil-and-grid subset an equilibrium keys on does not carry the epoch, since
    # an equilibrium does not depend on what lines the plasma is pointed at.
    assert tdu.subset("coils", "grid") == hhf.subset("coils", "grid")


def test_campaign_resolves_its_epoch():

    assert machine.epoch_of_campaign("OP1.1").key == "limiter"
    assert machine.epoch_of_campaign("OP1.2b").key == "tdu"
    assert machine.epoch_of_campaign("OP2.4").key == "hhf"
    with pytest.raises(KeyError):
        machine.epoch_of_campaign("OP9.9")


def test_only_the_cooled_epoch_carries_the_scraper():

    assert "scraper element" in machine.epoch("hhf").components
    assert "scraper element" not in machine.epoch("tdu").components
    assert machine.epoch("limiter").components == ()
    assert machine.epoch("hhf").cooled
    assert not machine.epoch("tdu").cooled


# -- impurity fits ----------------------------------------------------------

#: Largest step either fit takes across one of its temperature intervals. The published
#: coefficients are fitted interval by interval and are not constrained to join, so the
#: cooling rate steps by up to 7.2 % at 200 eV and the mean charge by 9.9 % at 3 eV. A
#: transcription slip in a coefficient moves a value by orders of magnitude, which this
#: bounds without asserting a continuity the fit does not have.
MAVRIN_LARGEST_STEP = 0.12


def test_mavrin_fits_step_no_further_than_published_across_their_intervals():
    for fit in (kinetics.CARBON_COOLING_RATE, kinetics.CARBON_MEAN_CHARGE):
        for edge in fit.t_max_ev[:-1]:
            below = float(fit(np.array([edge * (1 - 1e-6)]))[0])
            above = float(fit(np.array([edge * (1 + 1e-6)]))[0])
            assert abs(above - below) / abs(below) < MAVRIN_LARGEST_STEP, edge


def test_carbon_strips_fully_when_hot_and_recombines_when_cold():
    assert kinetics.mean_charge(np.array([2000.0]))[0] == pytest.approx(6.0, abs=0.05)
    assert kinetics.mean_charge(np.array([2.0]))[0] < 1.5


def test_cooling_rate_peaks_in_the_line_radiating_range():
    temperature = np.geomspace(1.0, 15000.0, 400)
    peak = temperature[int(np.argmax(kinetics.cooling_rate(temperature)))]
    assert 3.0 < peak < 30.0
    # And falls by two orders from that peak to fusion-relevant temperatures.
    assert kinetics.cooling_rate(np.array([3000.0]))[0] < 1e-2 * kinetics.cooling_rate(
        np.array([peak])
    )[0]


def test_composition_is_quasineutral_and_gives_the_exact_effective_charge():
    """At full stripping the answers are arithmetic, so they can be checked exactly."""
    n_e = np.array([8.0e19])
    parts = kinetics.composition(n_e, np.array([2000.0]), 0.02)
    charge = float(parts.charge[0])
    assert float(parts.ion_density_m3[0] + charge * parts.impurity_density_m3[0]) == (
        pytest.approx(float(n_e[0]))
    )
    expected = (1.0 - charge * 0.02) + 0.02 * charge**2
    assert float(parts.z_effective[0]) == pytest.approx(expected)


def test_no_carbon_leaves_the_plasma_pure():
    parts = kinetics.composition(np.array([8.0e19]), np.array([1000.0]), 0.0)
    assert float(parts.z_effective[0]) == pytest.approx(1.0)
    assert float(parts.dilution[0]) == pytest.approx(1.0)


def test_bremsstrahlung_scales_as_density_squared_and_root_temperature():
    one = kinetics.bremsstrahlung(
        np.array([1.0e20]), np.array([1000.0]), np.array([1.0e20])
    )[0]
    denser = kinetics.bremsstrahlung(
        np.array([2.0e20]), np.array([1000.0]), np.array([2.0e20])
    )[0]
    hotter = kinetics.bremsstrahlung(
        np.array([1.0e20]), np.array([4000.0]), np.array([1.0e20])
    )[0]
    assert denser == pytest.approx(4.0 * one)
    assert hotter == pytest.approx(2.0 * one)


# -- current diffusion ------------------------------------------------------

def test_resistive_time_matches_the_bessel_eigenvalue():
    """Uniform resistivity has an analytic answer, mu0 a^2 / (eta j01^2)."""
    a, eta = 0.5, 1.0e-7
    radius = np.linspace(1e-6, a, 41)
    expected = 4.0e-7 * np.pi * a**2 / (eta * current.FUNDAMENTAL_EIGENVALUE)
    assert current.resistive_time(
        radius, np.full_like(radius, eta)
    ) == pytest.approx(expected, rel=1e-3)


def test_current_diffusion_relaxes_to_the_bootstrap_profile():
    """The bootstrap profile is the steady state, and the current only rises toward it."""
    a = 0.49
    radius = np.linspace(1e-3, a, 51)
    eta = np.full_like(radius, 1.0e-7)
    target = 1.0e5 * (1.0 - (radius / a) ** 2)
    tau = current.resistive_time(radius, eta)
    evolved = current.evolve(
        radius, eta, target, np.linspace(0.0, 10.0 * tau, 300)
    )
    assert evolved.enclosed_current_a[0] == 0.0
    assert np.all(np.diff(evolved.enclosed_current_a) >= -1e-6)
    assert evolved.current_density_a_m2[-1] == pytest.approx(target, rel=5e-3)


def test_current_diffusion_holds_a_profile_already_at_steady_state():
    a = 0.49
    radius = np.linspace(1e-3, a, 51)
    eta = np.full_like(radius, 1.0e-7)
    target = 1.0e5 * (1.0 - (radius / a) ** 2)
    evolved = current.evolve(
        radius, eta, target, np.linspace(0.0, 1.0, 20), initial_a_m2=target
    )
    assert evolved.current_density_a_m2[-1] == pytest.approx(target, rel=5e-3)


# -- island detection -------------------------------------------------------

def _orbit(radius: float, transform: float, returns: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Poincare points of one trajectory on a circle of given radius and transform."""
    angle = 2.0 * np.pi * transform * np.arange(returns)
    return radius * np.cos(angle), radius * np.sin(angle)


def test_winding_recovers_a_transform_below_a_half():
    """Below half a turn per return the winding is the transform itself."""
    from w7x_twin.mhd import stepped_pressure

    r, z = _orbit(0.3, 0.21)
    assert stepped_pressure.net_winding(
        r[None, :][0], z[None, :][0], 0.0, 0.0
    ) == pytest.approx(0.21, rel=1e-6)


def test_winding_is_aliased_above_a_half():
    """Above it the unwrapping cannot tell the direction, so 1 - iota comes back."""
    from w7x_twin.mhd import stepped_pressure

    r, z = _orbit(0.3, 0.8333333)
    assert stepped_pressure.net_winding(r, z, 0.0, 0.0) == pytest.approx(
        1.0 - 0.8333333, rel=1e-6
    )


def test_island_is_the_run_of_trajectories_locked_to_the_resonance():
    """Nested surfaces sweep in transform; the ones inside an island do not."""
    from w7x_twin.mhd import stepped_pressure

    radii = np.linspace(0.20, 0.40, 41)
    resonance = 5.0 / 6.0
    # Trajectories between 0.28 and 0.32 are locked to the resonance, the rest sweep.
    transforms = np.where(
        (radii >= 0.28) & (radii <= 0.32), resonance, 0.80 + 0.25 * (radii - 0.20)
    )
    r = np.array([_orbit(radius, t)[0] for radius, t in zip(radii, transforms, strict=True)])
    z = np.array([_orbit(radius, t)[1] for radius, t in zip(radii, transforms, strict=True)])

    found = stepped_pressure.island_at_resonance(r, z, 0.0, 0.0, resonance)
    assert found["trajectories"] >= 8
    assert found["width_m"] == pytest.approx(0.04, abs=0.006)


def test_no_island_where_nothing_is_locked():
    from w7x_twin.mhd import stepped_pressure

    radii = np.linspace(0.20, 0.40, 41)
    transforms = 0.60 + 0.25 * (radii - 0.20)
    r = np.array([_orbit(radius, t)[0] for radius, t in zip(radii, transforms, strict=True)])
    z = np.array([_orbit(radius, t)[1] for radius, t in zip(radii, transforms, strict=True)])
    found = stepped_pressure.island_at_resonance(r, z, 0.0, 0.0, 5.0 / 6.0)
    assert found["width_m"] == 0.0
    assert found["trajectories"] == 0


# -- scrape-off layer -------------------------------------------------------

def test_equal_power_and_momentum_loss_leaves_the_target_temperature_alone():
    """The two enter only through their ratio, which is why detachment scans power."""
    from w7x_twin.plasma import edge

    attached = edge.solve_two_point_extended(1e19, 5e7, 100.0)
    both = edge.solve_two_point_extended(
        1e19, 5e7, 100.0, power_loss=0.6, momentum_loss=0.6
    )
    assert both.target_temperature_ev == pytest.approx(
        attached.target_temperature_ev, rel=1e-3
    )
    # And the density falls by exactly the momentum loss.
    assert both.target_density_m3 == pytest.approx(
        0.4 * attached.target_density_m3, rel=1e-3
    )


def test_power_loss_alone_cools_the_target():
    from w7x_twin.plasma import edge

    attached = edge.solve_two_point_extended(1e19, 5e7, 100.0)
    lossy = edge.solve_two_point_extended(
        1e19, 5e7, 100.0, power_loss=0.9
    )
    assert lossy.target_temperature_ev < 0.5 * attached.target_temperature_ev


def test_two_point_conserves_the_sheath_condition():
    """The flux the sheath carries has to equal the flux supplied to it."""
    from w7x_twin.plasma import edge

    solution = edge.solve_two_point(2e19, 5e7, 100.0)
    carried = (
        edge.SHEATH_TRANSMISSION
        * solution.target_density_m3
        * solution.sound_speed_m_s
        * solution.target_temperature_ev
        * edge.ELEMENTARY_CHARGE
    )
    assert carried == pytest.approx(5e7, rel=1e-3)


def _analysis(name):
    """Import an analyses module, so a test can check the code a result was produced by."""
    import importlib

    return importlib.import_module(f"w7x_twin.analyses.{name}")


# ---------------------------------------------------------------------------
# One drift-kinetic diffusivity model
# ---------------------------------------------------------------------------


def _synthetic_profile():
    """A two-surface drift-kinetic scan with a plain power law in collisionality."""
    nu = np.geomspace(1e-5, 1e-1, 8)
    tables = []
    for s, level in ((0.1, 3.0e-2), (0.5, 1.0e-2)):
        tables.append(
            neoclassical.MonoenergeticCoefficients(
                s=s,
                collisionality=nu,
                radial_field=np.zeros_like(nu),
                d11=level / nu,
                d31=np.zeros_like(nu),
                d33=np.full_like(nu, 0.4),
                d33_spitzer=np.full_like(nu, 1.0),
            )
        )
    return neoclassical.MonoenergeticProfile(
        surfaces=np.array([t.s for t in tables]), tables=tuple(tables)
    )


def test_the_two_named_surfaces_are_kept_apart():
    """The reporting surface and the single-surface table's surface are different numbers."""
    assert neoclassical.REFERENCE_SURFACE != neoclassical.SINGLE_SURFACE
    # surface_tables uses its reference only when a single table stands in, so its default
    # has to be that table's surface and not the reporting one.
    import inspect

    default = inspect.signature(neoclassical.surface_tables).parameters[
        "reference_surface"
    ].default
    assert default == neoclassical.SINGLE_SURFACE


def test_ripple_scaling_uses_the_reference_it_is_given():
    """A single-surface table is carried to another surface by the ripple to the 3/2."""
    nu = np.geomspace(1e-5, 1e-1, 5)
    table = neoclassical.MonoenergeticCoefficients(
        s=neoclassical.SINGLE_SURFACE,
        collisionality=nu,
        radial_field=np.zeros_like(nu),
        d11=1.0 / nu,
        d31=np.zeros_like(nu),
        d33=np.full_like(nu, 0.4),
        d33_spitzer=np.full_like(nu, 1.0),
    )
    ripple = neoclassical.EffectiveRipple(
        s=np.array([0.1, 0.2, 0.4]),
        rho=np.sqrt(np.array([0.1, 0.2, 0.4])),
        eps_32=np.array([1.0e-3, 2.0e-3, 4.0e-3]),
    )
    (scaled, weight), = neoclassical.surface_tables(
        table, 0.4, which="d11", ripple=ripple,
        reference_surface=neoclassical.SINGLE_SURFACE,
    )
    assert weight == 1.0
    # Both surfaces are stored points, so no interpolation intervenes and the ratio is the
    # stored one exactly.
    assert np.allclose(scaled.d11 / table.d11, 4.0e-3 / 2.0e-3)
    assert np.allclose(
        (ripple.at(0.4) / ripple.at(0.2)) ** 1.5, 4.0e-3 / 2.0e-3
    )
    # At the reference itself the scaling is the identity.
    (same, _), = neoclassical.surface_tables(
        table, neoclassical.SINGLE_SURFACE, which="d11", ripple=ripple,
        reference_surface=neoclassical.SINGLE_SURFACE,
    )
    assert np.allclose(same.d11, table.d11)


def test_the_module_diffusivity_matches_the_script_it_was_taken_from():
    """The packaged model returns what the transport entry's own builder returns."""
    profile = _synthetic_profile()
    ripple = neoclassical.EffectiveRipple(
        s=np.array([0.1, 0.5]),
        rho=np.sqrt(np.array([0.1, 0.5])),
        eps_32=np.array([1.0e-3, 2.0e-3]),
    )
    s = np.linspace(0.05, 0.9, 9)
    temperature = 3000.0 * (1.0 - s) + 100.0
    density = 5.0e19 * (1.0 - 0.5 * s)

    original = _analysis("plasma").build_neoclassical(
        profile, ripple, radial_field_v_m=0.0, minor_radius=0.49
    )
    packaged = neoclassical.diffusivity_model(
        profile, ripple, 0.49, radial_field_v_m=0.0,
        ion_fraction=_analysis("plasma").ION_FRACTION,
    )
    assert np.allclose(packaged(s, temperature, density), original(s, temperature, density),
                       rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Comparing a sampled filament with a Fourier curve
# ---------------------------------------------------------------------------


def _refine(points, samples):
    from w7x_twin.magnetics import field

    return field.refine(points, samples)


def test_refinement_reproduces_a_curve_its_samples_came_from():
    """Interpolating a sampled closed curve returns the curve, not a polyline through it."""
    def curve(t):
        return np.stack(
            [
                3.0 + 0.5 * np.cos(t) + 0.05 * np.cos(3 * t),
                0.5 * np.sin(t) - 0.03 * np.sin(2 * t),
                0.2 * np.sin(t) + 0.01 * np.cos(5 * t),
            ],
            axis=1,
        )

    coarse = curve(np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False))
    dense = curve(np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False))
    refined = _refine(coarse, 4096)
    assert np.abs(refined - dense).max() < 1e-10
    # The polyline the filament would be compared as sits measurably off the curve, which
    # is the error this interpolation removes.
    gaps = np.linalg.norm(np.diff(coarse, axis=0, append=coarse[:1]), axis=1)
    assert gaps.max() > 1e-3


# ---------------------------------------------------------------------------
# The measured n = 1 trim waveform
# ---------------------------------------------------------------------------


def test_trim_waveform_is_a_pure_n_equals_one():
    """Five currents on a cosine of one full turn sum to zero and have no n = 0 part."""
    currents = np.array(list(machine.trim_waveform(98.0, 162.0).values()))
    assert len(currents) == machine.NUM_FIELD_PERIODS
    assert currents.sum() == pytest.approx(0.0, abs=1e-9 * np.abs(currents).max())
    spectrum = np.abs(np.fft.rfft(currents))
    assert np.argmax(spectrum) == 1
    assert spectrum[0] < 1e-9 * spectrum[1]


def test_trim_phase_places_the_peak_on_the_named_module():
    """A phase of zero puts the maximum on module one, which is where it is measured from."""
    currents = machine.trim_waveform(100.0, 0.0)
    assert currents["trim_a1"] == pytest.approx(100.0)
    assert max(currents.values()) == pytest.approx(100.0)
    # Turning the phase by one module's worth moves the peak by one module.
    stepped = machine.trim_waveform(100.0, 360.0 / machine.NUM_FIELD_PERIODS)
    assert stepped["trim_a2"] == pytest.approx(100.0)


def test_trim_amplitude_scales_the_waveform_linearly():
    """The waveform is a current, so twice the amplitude is twice every coil."""
    one = machine.trim_waveform(50.0, 37.0)
    two = machine.trim_waveform(100.0, 37.0)
    for key, value in one.items():
        assert two[key] == pytest.approx(2.0 * value)


def test_measured_trim_settings_carry_a_source_and_a_known_configuration():
    """Every published setting names its configuration, its method and its source."""
    from w7x_twin.records import programmes

    known = set(machine.all_keys())
    for setting in programmes.TRIM_SETTINGS:
        assert setting.method and setting.source
        assert 0.0 < setting.amplitude_a < 2000.0
        # 'high_iota' is a published configuration this library does not carry; the rest
        # have to resolve, or a setting would be applied to a machine state that is not
        # the one it was measured in.
        if setting.configuration != "high_iota":
            assert setting.configuration in known


def test_conductivity_reduction_is_below_unity_when_d33_is():
    """Trapped particles reduce the parallel conductivity, so the ratio is under one."""
    nu = np.geomspace(1e-6, 1e-1, 6)
    table = neoclassical.MonoenergeticCoefficients(
        s=0.2,
        collisionality=nu,
        radial_field=np.zeros_like(nu),
        d11=np.full_like(nu, 1.0),
        d31=np.zeros_like(nu),
        d33=np.full_like(nu, 0.4),
        d33_spitzer=np.full_like(nu, 1.0),
    )
    ratio = current.conductivity_reduction(table, 8.0e19, 2000.0)
    assert ratio == pytest.approx(0.4, rel=1e-6)


# -- target incidence -------------------------------------------------------

def _slab(dr_dphi: float, dz_dphi: float, num_cuts: int = 5):
    """A flat target whose contour drifts linearly with toroidal angle."""
    from w7x_twin.hardware.walls import Component

    diagonal = 0.5 * np.sqrt(2.0)
    phi = np.linspace(0.0, np.deg2rad(2.0), num_cuts)
    u = np.linspace(-0.1, 0.1, 9)
    r = 5.5 + diagonal * u[None, :] + dr_dphi * phi[:, None]
    z = 0.5 + diagonal * u[None, :] + dz_dphi * phi[:, None]
    return Component(name="slab", phi=phi, r=r, z=z)


def test_surface_normal_reduces_to_the_poloidal_one_when_the_contour_does_not_move():
    """A genuinely swept contour gives the poloidal normal, so both routes agree."""
    from w7x_twin.plasma import edge
    from w7x_twin.hardware import walls

    swept_component = _slab(0.0, 0.0)
    r = np.array([5.50, 5.53])
    z = np.array([0.50, 0.50])
    phi = np.array([np.deg2rad(0.5), np.deg2rad(1.5)])

    def field(r_, phi_, z_):
        return (np.full_like(r_, 0.03), np.full_like(r_, 1.0), np.full_like(r_, 0.07))

    frame = walls.surface_frame(swept_component, r, z, phi)
    assert frame["dr_dphi"] == pytest.approx(0.0, abs=1e-12)
    assert frame["dz_dphi"] == pytest.approx(0.0, abs=1e-12)

    tangent_r, tangent_z = edge.contour_tangent(swept_component, r, z, phi)
    poloidal = edge.incidence_sine(field, r, phi, z, tangent_r, tangent_z)
    surface = edge.surface_incidence_sine(field, r, phi, z, frame)
    assert surface == pytest.approx(poloidal, rel=1e-12)


def test_a_moving_contour_changes_the_incidence_it_reports():
    """The toroidal derivative is what carries the inclination, so it must move the answer."""
    from w7x_twin.plasma import edge
    from w7x_twin.hardware import walls

    r = np.array([5.50])
    z = np.array([0.50])
    phi = np.array([np.deg2rad(1.0)])

    def field(r_, phi_, z_):
        return (np.full_like(r_, 0.03), np.full_like(r_, 1.0), np.full_like(r_, 0.07))

    def sine(dr_dphi: float) -> float:
        component = _slab(dr_dphi, 0.0)
        frame = walls.surface_frame(component, r, z, phi)
        return float(
            edge.surface_incidence_sine(field, r, phi, z, frame)[0]
        )

    assert sine(0.0) != pytest.approx(sine(1.5), rel=1e-3)
    # A steeper drift turns the surface towards the poloidal plane, so an almost toroidal
    # field arrives on it less obliquely.
    assert sine(3.0) > sine(1.5) > sine(0.0)


def test_incidence_is_unchanged_under_the_stellarator_symmetry_that_builds_the_lower_units():
    """A lower unit is the (phi, Z) -> (-phi, -Z) image, so a mirrored strike matches."""
    from w7x_twin.plasma import edge
    from w7x_twin.hardware import walls

    component = _slab(1.5, 0.8)
    r = np.array([5.52])
    z = np.array([0.50])
    phi = np.array([np.deg2rad(1.0)])

    def field(r_, phi_, z_):
        return (np.full_like(r_, 0.03), np.full_like(r_, 1.0), np.full_like(r_, 0.07))

    def mirrored(r_, phi_, z_):
        b_r, b_phi, b_z = field(r_, -phi_, -z_)
        return (b_r, -b_phi, -b_z)

    upper = edge.surface_incidence_sine(
        field, r, phi, z, walls.surface_frame(component, r, z, phi)
    )
    lower = edge.surface_incidence_sine(
        mirrored, r, -phi, -z, walls.surface_frame(component, r, -z, -phi)
    )
    assert lower == pytest.approx(upper, rel=1e-12)


def test_the_divertor_targets_are_cut_at_the_published_element_width():
    """The target files are cut at 0.500 degrees, the toroidal extent of one element."""
    from pathlib import Path

    from w7x_twin.hardware import walls

    if not Path("data/pfc").is_dir():
        pytest.skip("component files not present")
    elements = walls.load_components("data/pfc")
    targets = [e for e in elements if e.name in walls.TARGET_COMPONENTS]
    assert targets, "no divertor target loaded"
    for element in targets:
        step = np.abs(np.diff(element.phi))
        assert np.degrees(step) == pytest.approx(0.5, abs=1e-6)


def test_the_n1_harmonic_of_a_helical_shift_is_taken_componentwise():
    """A helical shift has constant magnitude, so its magnitude carries no n = 1 at all."""
    from w7x_twin.magnetics.field import n1_amplitude

    planes = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    amplitude = 0.0123
    shift = np.stack(
        [amplitude * np.cos(planes), amplitude * np.sin(planes)], axis=-1
    )
    # The magnitude is constant, so a harmonic of it would find nothing.
    assert np.allclose(np.hypot(shift[:, 0], shift[:, 1]), amplitude)
    magnitude_harmonic = np.abs(
        np.fft.rfft(np.hypot(shift[:, 0], shift[:, 1])) / len(shift)
    )[1] * 2.0
    assert magnitude_harmonic == pytest.approx(0.0, abs=1e-12)
    assert n1_amplitude(shift) == pytest.approx(amplitude, rel=1e-9)


def test_the_periodic_floor_tolerance_brackets_roundoff_and_a_real_asymmetry():
    """A periodic field makes the five module loads identical to round-off, not to zero."""
    module = _analysis("discharges")

    equal = np.full(5, 3.2e5)
    # Perturb by one bit each way, which is what summing weights over traced lines does.
    equal = np.nextafter(equal, [0.0, np.inf, 0.0, np.inf, 0.0])
    roundoff = float(equal.std() / equal.mean())
    assert roundoff > 0.0, "the point of the test is that this is not exactly zero"
    assert roundoff < module.PERIODIC_FLOOR_TOLERANCE
    assert module.PERIODIC_FLOOR_TOLERANCE < 0.076


def test_a_harmonic_difference_is_the_magnitude_of_a_difference():
    """The field a deviation adds is |A - B|, not |A| - |B|."""
    from w7x_twin.magnetics import field as source

    points = 256
    phi = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    leakage, perturbation = 0.16e-3, 0.23e-3

    def circle(values):
        class Field:
            def __call__(self, r, p, z):
                return (np.interp(p, phi, values), np.zeros_like(p), np.zeros_like(p))
        return Field()

    # A perturbation a quarter period out of phase with what is already on the circle.
    base = leakage * np.cos(phi)
    added = base + perturbation * np.sin(phi)
    before = source.radial_harmonics(circle(base), points)
    after = source.radial_harmonics(circle(added), points)

    assert np.iscomplexobj(before), "the harmonics have to stay complex to be differenced"
    assert abs(after[1] - before[1]) == pytest.approx(perturbation, rel=1e-6)
    # Subtracting magnitudes returns hypot(a, b) - a for orthogonal phasors, which
    # understates the perturbation and depends on the leakage it is measured against.
    wrong = abs(abs(after[1]) - abs(before[1]))
    assert wrong == pytest.approx(
        np.hypot(leakage, perturbation) - leakage, rel=1e-6
    )
    assert wrong < perturbation


def _band(drift_m_per_rad: float, local_sigma: float, mirror: bool = False):
    """A synthetic strike band of known local width on a flat slab target."""
    from w7x_twin.hardware.walls import Component

    rng = np.random.default_rng(7)
    count = 4000
    num_cuts = 21
    phi = np.linspace(0.0, np.deg2rad(10.0), num_cuts)
    stations = np.linspace(0.0, 0.30, 61)
    component = Component(
        name="slab",
        phi=phi,
        r=np.tile(5.0 + stations, (num_cuts, 1)),
        z=np.tile(np.full_like(stations, 0.5), (num_cuts, 1)),
    )

    class Strikes:
        pass

    strike_phi = rng.uniform(0.0, np.deg2rad(10.0), count)
    arc = 0.10 + drift_m_per_rad * (strike_phi - np.deg2rad(5.0))
    arc = arc + rng.normal(0.0, local_sigma, count)
    strikes = Strikes()
    strikes.struck = np.ones(count, dtype=bool)
    strikes.r = 5.0 + np.clip(arc, 0.0, 0.30)
    strikes.z = np.full(count, -0.5 if mirror else 0.5)
    strikes.phi = -strike_phi if mirror else strike_phi
    strikes.component = np.zeros(count, dtype=int)
    strikes.connection_length_m = np.full(count, 50.0)
    strikes.start_r = np.full(count, 6.0)
    return component, strikes


def test_the_deskewed_profile_recovers_the_local_band_width():
    """The drift along the target is coordinate, not width, and deskew removes it."""
    from w7x_twin.plasma import edge

    local = np.sqrt(2.0 * np.pi) * 0.010
    frame = {"slab": (0.0, False, 0.30)}
    component, strikes = _band(0.35, 0.010)
    raw = edge.target_profile(
        strikes, [component], frame, np.ones(strikes.r.shape), 1.0e6
    )["peak_integral_width_m"]
    fixed = edge.target_profile(
        strikes, [component], frame, np.ones(strikes.r.shape), 1.0e6, deskew=True
    )
    assert raw > 1.5 * local
    assert fixed["peak_integral_width_m"] == pytest.approx(local, rel=0.12)
    assert fixed["per_element"]["slab"]["band_drift_m_per_rad"] == pytest.approx(
        0.35, rel=0.05
    )
    component, flat = _band(0.0, 0.010)
    untouched = edge.target_profile(
        flat, [component], frame, np.ones(flat.r.shape), 1.0e6, deskew=True
    )
    assert untouched["peak_integral_width_m"] == pytest.approx(local, rel=0.12)


def test_the_deskew_treats_a_lower_unit_as_the_mirror_it_is():
    """The lower units are the (phi, Z) -> (-phi, -Z) image, so their drift mirrors."""
    from w7x_twin.plasma import edge

    local = np.sqrt(2.0 * np.pi) * 0.010
    frame = {"slab": (0.0, False, 0.30)}
    component, upper = _band(0.35, 0.010)
    _, lower = _band(0.35, 0.010, mirror=True)

    class Strikes:
        pass

    mixed = Strikes()
    for name in ("struck", "r", "z", "phi", "component", "connection_length_m", "start_r"):
        setattr(
            mixed, name,
            np.concatenate([getattr(upper, name), getattr(lower, name)]),
        )
    fixed = edge.target_profile(
        mixed, [component], frame, np.ones(mixed.r.shape), 1.0e6, deskew=True
    )
    assert fixed["peak_integral_width_m"] == pytest.approx(local, rel=0.12)


def test_the_power_weighted_connection_length_sits_inside_the_distribution():
    """The average is over the power each line carries, not over the lines."""
    from w7x_twin.plasma import edge

    lengths = np.array([2.0, 4.0, 8.0, 190.0])
    # All the power on the longest line, then all of it on the shortest.
    assert edge.power_weighted_connection_length(
        lengths, np.array([0.0, 0.0, 0.0, 1.0])
    ) == pytest.approx(190.0)
    assert edge.power_weighted_connection_length(
        lengths, np.array([1.0, 0.0, 0.0, 0.0])
    ) == pytest.approx(2.0)
    # Spread evenly it is the plain mean, and always inside the distribution.
    even = edge.power_weighted_connection_length(
        lengths, np.ones_like(lengths)
    )
    assert even == pytest.approx(lengths.mean())
    assert lengths.min() <= even <= lengths.max()
    # No power anywhere is not a length.
    assert np.isnan(
        edge.power_weighted_connection_length(lengths, np.zeros_like(lengths))
    )


# -- heating, neutrals, particles, evolution ---------------------------------

def test_the_x2_cutoff_is_the_r_wave_one_not_the_upper_hybrid():
    """At the h-th harmonic the R-wave cutoff leaves w_pe^2 = (1 - 1/h) w^2."""
    from w7x_twin.plasma import transport

    assert transport.cutoff_density_m3() == pytest.approx(1.2e20, rel=0.02)
    # The first harmonic in X-mode has no accessible window at all.
    assert transport.cutoff_density_m3(harmonic=1) == 0.0
    # And the resonant field for X2 at 140 GHz is the machine's own 2.5 T.
    assert transport.resonant_field_t() == pytest.approx(2.5, rel=0.01)


def test_a_stepped_waveform_steps_rather_than_ramps():
    """Each segment holds its own power, so the value just after a step is the new one."""
    from w7x_twin.plasma import current

    waveform = current.Waveform.steps(((1.7, 2.0e6), (4.3, 3.4e6)))
    assert waveform.at(1.0) == pytest.approx(2.0e6)
    assert waveform.at(1.69) == pytest.approx(2.0e6)
    assert waveform.at(1.8) == pytest.approx(3.4e6)
    assert waveform.at(6.0) == pytest.approx(3.4e6)


def test_the_beam_gives_more_of_itself_to_ions_in_a_colder_plasma():
    """The critical energy goes with the temperature, so a cold plasma is ion-heated."""
    from w7x_twin.plasma import transport

    cold, hot = transport.beam_power_split(55.0e3, np.array([500.0, 5000.0]))[0]
    assert 0.0 < cold < hot < 1.0


def test_the_energy_relaxes_to_power_times_confinement_time():
    """A flat waveform run long enough has to reach W = P tau_E and stay there."""
    from w7x_twin.plasma import current

    power, tau = 5.0e6, 0.15
    trace = current.advance(
        current.Waveform.steps(((5.0, power),)),
        confinement_time=lambda energy, watts: tau,
        bootstrap_current=lambda energy: 0.0,
        edge_transform=lambda current: 1.0,
        minor_radius_m=0.5,
        steps=2001,
    )
    assert trace.stored_energy_j[-1] == pytest.approx(power * tau, rel=1e-3)
    assert np.all(trace.stored_energy_j >= 0.0)


def test_the_recycling_pressure_rises_with_the_flux_that_feeds_it():
    """The neutral density is the ion flux divided by the speed it leaves at."""
    from w7x_twin.plasma import edge

    low = edge.recycling_layer(1.0e23, 20.0, 5.0e18)
    high = edge.recycling_layer(2.0e23, 20.0, 5.0e18)
    assert high.neutral_density_m3 == pytest.approx(2.0 * low.neutral_density_m3)
    assert high.pressure_pa == pytest.approx(2.0 * low.pressure_pa)
    # The mean free path is set by the plasma it penetrates, not by the flux.
    assert high.mean_free_path_m == pytest.approx(low.mean_free_path_m)


def test_a_deeper_particle_source_gives_a_more_peaked_density():
    """Where the particles enter is what sets the peaking, which is the point of the balance."""
    from w7x_twin.plasma import transport

    s = np.linspace(0.0, 1.0, 121)
    edge = transport.peaking_for_source(0.95, s, 0.49, 5.0e18, 1.0e22)
    core = transport.peaking_for_source(0.20, s, 0.49, 5.0e18, 1.0e22)
    assert core.peaking > edge.peaking
    assert np.all(core.density_m3 > 0.0)


def test_the_carbon_profile_reaches_its_edge_fraction_at_the_separatrix():
    """The source is at the target, so the fraction is pinned at the edge and rises inward."""
    from w7x_twin.plasma import kinetics

    s = np.linspace(0.0, 1.0, 81)
    density = np.full_like(s, 5.0e19)
    fraction = kinetics.carbon_profile(s, density, 0.49, 0.02)
    assert fraction[-1] == pytest.approx(0.02, rel=1e-6)
    # An inward pinch accumulates it, so the core fraction is the larger one.
    assert fraction[0] > fraction[-1]


def test_every_data_path_the_package_names_exists():
    """A module rename must not rewrite a data filename."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if not (root / "data").is_dir():
        pytest.skip("data directory not present")
    named = set()
    for path in (root / "src").rglob("*.py"):
        named.update(re.findall(r'"(data/[A-Za-z_0-9.\-]+)"', path.read_text(encoding="utf-8")))
    assert named, "no data paths found to check"
    missing = sorted(p for p in named if not (root / p).exists())
    assert not missing, f"named but absent: {missing}"


def test_every_module_invoked_as_a_subprocess_still_exists():
    """A `python -m` target must be an importable module, and its path must resolve."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    invoked = set()
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        invoked.update(re.findall(r'"-m",\s*"([A-Za-z_][A-Za-z_0-9.]*)"', text))
    assert invoked, "no subprocess module targets found to check"
    missing = [name for name in invoked
               if not (source / Path(*name.split("."))).with_suffix(".py").exists()]
    assert not missing, f"invoked with -m but absent: {missing}"


# -- the substitutions into the main path ------------------------------------

def test_a_target_sourced_carbon_profile_is_peaked_and_a_flat_one_is_not():
    """Carbon enters at the targets, so its fraction is a profile and not one number."""
    from w7x_twin.plasma import kinetics
    import dataclasses

    flat = dataclasses.replace(kinetics.HIGH_PERFORMANCE, carbon_fraction=0.02)
    carried = dataclasses.replace(flat, carbon_from_target=True)
    s = np.linspace(0.0, 1.0, 41)
    assert np.allclose(flat.carbon_profile(s), 0.02)
    profile = carried.carbon_profile(s)
    assert profile[-1] == pytest.approx(0.02, rel=1e-6)
    # The zero-flux solution accumulates carbon inward, so its density rises; the fraction
    # is that against an electron density which rises far faster, so the fraction falls.
    density = carried.density(s)
    carbon = profile * density
    assert carbon[0] > carbon[-1]
    assert profile[0] < profile[-1]
    # With no carbon at all neither route invents any.
    assert np.all(kinetics.HIGH_PERFORMANCE.carbon_profile(s) == 0.0)


def test_the_effective_charge_follows_the_carbon_profile_rather_than_one_number():
    """A radial carbon fraction makes the effective charge a profile, not a constant."""
    from w7x_twin.plasma import kinetics
    import dataclasses

    s = np.array([0.0, 0.5, 1.0])
    flat = dataclasses.replace(kinetics.HIGH_PERFORMANCE, carbon_fraction=0.02)
    carried = dataclasses.replace(flat, carbon_from_target=True)
    assert np.all(carried.z_effective_profile(s) > 1.0)
    # The flat fraction gives the same dilution everywhere the temperature is the same;
    # the carried one varies with the electron profile it is measured against.
    assert carried.z_effective_profile(s)[0] != pytest.approx(
        flat.z_effective_profile(s)[0], rel=1e-3
    )
    assert carried.z_effective_profile(s)[-1] == pytest.approx(
        flat.z_effective_profile(s)[-1], rel=1e-3
    )


def test_a_second_ion_species_raises_the_mean_mass_and_slows_the_sound_speed():
    """A heavier fuel is slower at the same temperature."""
    from w7x_twin.plasma import edge
    from w7x_twin.plasma import kinetics
    import dataclasses

    s = np.array([0.0, 0.5])
    pure = kinetics.HIGH_PERFORMANCE
    assert np.allclose(pure.mean_ion_mass_amu(s), 1.0)
    mixed = dataclasses.replace(pure, second_ion_fraction=0.5, second_ion_mass_amu=2.0)
    assert np.allclose(mixed.mean_ion_mass_amu(s), 1.5)
    assert edge.sound_speed(np.array([50.0]), 1.5)[0] < edge.sound_speed(np.array([50.0]))[0]
    assert edge.sound_speed(np.array([50.0]), 1.0)[0] == pytest.approx(
        edge.sound_speed(np.array([50.0]))[0]
    )


def test_a_computed_deposition_replaces_the_gaussian_it_is_given_instead_of():
    """Supplying an absorption profile makes the heating follow it, not the width."""
    from w7x_twin.plasma import transport

    s = np.linspace(0.0, 1.0, 81)
    # An absorption layer at mid-radius, nothing like the on-axis Gaussian.
    profile = np.exp(-0.5 * ((s - 0.7) / 0.05) ** 2)
    deposition = transport.Deposition(
        s=s, profile_w=profile, absorbed_fraction=1.0, peak_s=0.7
    )
    computed = transport.Heating.from_deposition(5.0e6, deposition)
    gaussian = transport.Heating(power_w=5.0e6)
    probe = np.array([0.0, 0.7])
    assert computed.profile(probe)[1] > computed.profile(probe)[0]
    assert gaussian.profile(probe)[0] > gaussian.profile(probe)[1]


def test_the_mixing_length_constant_comes_from_saturated_runs_only():
    """A nonlinear trace that never saturated cannot fix the constant."""
    import json
    from pathlib import Path

    record = Path(__file__).resolve().parents[1] / "results/turbulence/mixing_length_constant.json"
    if not record.is_file():
        pytest.skip("no mixing-length record")
    stored = json.loads(record.read_text())
    points = stored.get("points")
    assert points, "the record carries its points under 'points'"
    saturated = [p for p in points if p.get("nonlinear_saturation_state") == "saturated"]
    assert saturated, "no saturated point to take a constant from"
    for point in saturated:
        assert np.isfinite(point["constant"]) and point["constant"] > 0.0


def test_the_anomalous_channel_returns_finite_diffusivities():
    """The channel feeds the power balance, so a NaN there becomes a NaN stored energy."""
    from w7x_twin.plasma import transport

    channel = transport.anomalous_channel(2.31, 0.4922)
    if channel is None:
        pytest.skip("no growth-rate grid or constant on disk")
    s = np.linspace(0.05, 0.95, 41)
    temperature = 3000.0 * (1.0 - s) + 100.0
    density = np.full_like(s, 6.0e19)
    chi = np.asarray(channel(s, temperature, density), dtype=float)
    assert np.all(np.isfinite(chi)), "the channel produced a non-finite diffusivity"
    assert np.all(chi >= 0.0)
    # a/L_T of this profile crosses the threshold near the axis and is far above it over
    # the middle of the profile, where the channel has to be alive.
    gradient = -0.4922 * np.gradient(np.log(temperature), 0.4922 * np.sqrt(s))
    above = gradient > 3.0
    assert above.any()
    assert np.all(chi[above] > 0.0)


def test_the_global_mode_operator_is_positive_without_a_drive():
    """Field-line bending alone cannot destabilise anything, so every eigenvalue is positive."""
    from w7x_twin.mhd import diagnostics

    s = np.linspace(0.0, 1.0, 61)
    iota = 0.85 + 0.11 * s
    quiet = diagnostics.global_modes(
        s, np.zeros_like(s), iota, np.zeros_like(s),
        minor_radius_m=0.49, major_radius_m=5.5, field_t=2.31,
        modes=((1, 1), (2, 2), (3, 3), (2, 1)),
    )
    assert quiet, "no modes were formed"
    for mode in quiet:
        assert mode.eigenvalue > 0.0, f"{mode.n}/{mode.m} unstable on bending alone"
        assert not mode.unstable


def _mercier_like(d_merc, d_curr, d_shear):
    import types

    arrays = [np.asarray(a, dtype=float) for a in (d_merc, d_curr, d_shear)]
    return types.SimpleNamespace(
        s=np.linspace(0.0, 1.0, len(arrays[0])),
        DMerc=arrays[0], Dcurr=arrays[1], Dshear=arrays[2],
    )


def test_resistive_interchange_reduces_to_mercier_without_a_cross_term():
    """With no Pfirsch-Schlueter cross term the two criteria are the same criterion."""
    from w7x_twin.mhd import diagnostics

    result = diagnostics.resistive_interchange(
        _mercier_like([0.0, 0.02, 0.01, 0.0], np.zeros(4), np.full(4, 0.5))
    )
    assert np.allclose(result.d_r, [0.0, 0.02, 0.01, 0.0])
    assert np.allclose(result.h, 0.0)


def test_resistive_interchange_at_half_h_loses_exactly_the_shear_term():
    """At H = 1/2 the resistive criterion is Mercier with the shear stabilisation removed."""
    from w7x_twin.mhd import diagnostics

    d_shear = np.array([0.3, 0.5, 0.8, 0.2])
    result = diagnostics.resistive_interchange(
        _mercier_like(np.full(4, 0.1), -2.0 * d_shear, d_shear)
    )
    assert np.allclose(result.h, 0.5)
    assert np.allclose(result.d_r, 0.1 - d_shear)


def test_resistive_interchange_is_stricter_exactly_inside_the_ggj_window():
    """D_R falls below DMerc for 0 < H < 1 and not outside, since D_R - D_I = H - H^2."""
    from w7x_twin.mhd import diagnostics

    d_shear = np.full(5, 0.25)
    h = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    result = diagnostics.resistive_interchange(
        _mercier_like(np.zeros(5), -4.0 * d_shear * h, d_shear)
    )
    assert np.allclose(result.h, h)
    assert result.d_r[2] < 0.0
    assert result.d_r[0] > 0.0 and result.d_r[4] > 0.0
    assert result.d_r[1] == pytest.approx(0.0) and result.d_r[3] == pytest.approx(0.0)


def test_resistive_interchange_has_no_answer_on_a_shearless_surface():
    from w7x_twin.mhd import diagnostics

    result = diagnostics.resistive_interchange(
        _mercier_like([0.1, 0.1], [0.01, 0.01], [0.5, 0.0])
    )
    assert np.isfinite(result.d_r[0])
    assert np.isnan(result.d_r[1]) and np.isnan(result.h[1])


def test_tearing_growth_carries_the_fkr_exponents():
    """gamma scales as delta'^(4/5), eta^(3/5) and (k' v_A)^(2/5), and marginal is quiet."""
    from w7x_twin.mhd import diagnostics

    base = diagnostics.tearing_growth_rate(2.0, 1e-7, 0.5, 4e6)
    assert base > 0.0
    assert diagnostics.tearing_growth_rate(4.0, 1e-7, 0.5, 4e6) == pytest.approx(
        base * 2.0**0.8
    )
    assert diagnostics.tearing_growth_rate(2.0, 2e-7, 0.5, 4e6) == pytest.approx(
        base * 2.0**0.6
    )
    assert diagnostics.tearing_growth_rate(2.0, 1e-7, 1.0, 4e6) == pytest.approx(
        base * 2.0**0.4
    )
    assert diagnostics.tearing_growth_rate(0.0, 1e-7, 0.5, 4e6) == 0.0
    assert diagnostics.tearing_growth_rate(-3.0, 1e-7, 0.5, 4e6) == 0.0
    assert diagnostics.tearing_growth_rate(float("nan"), 1e-7, 0.5, 4e6) == 0.0


def test_a_well_stabilises_the_global_mode_and_a_hill_destabilises_it():
    """The drive must reach the eigenvalue, with the sign the package already uses."""
    from w7x_twin.mhd import diagnostics

    s = np.linspace(0.0, 1.0, 61)
    iota = 0.85 + 0.11 * s
    kept = dict(minor_radius_m=0.49, major_radius_m=5.5, field_t=2.31, modes=((1, 1),))

    def lowest(well, peak_pa):
        return diagnostics.global_modes(
            s, peak_pa * (1.0 - s) ** 2, iota, np.full_like(s, well), **kept
        )[0].eigenvalue

    quiet, driven = 1.0e3, 8.0e4
    assert lowest(0.0, quiet) == pytest.approx(lowest(0.0, driven), rel=1e-12)
    rise = lowest(+0.5, driven) - lowest(+0.5, quiet)
    fall = lowest(-0.5, driven) - lowest(-0.5, quiet)
    assert rise > 0.0 > fall
    assert rise == pytest.approx(-fall, rel=1e-3)


def test_the_control_coils_can_carry_an_n_equals_two_pattern():
    """Five coils driven as one circuit can only be periodic, so the 2/2 needs them apart."""
    from w7x_twin.hardware import machine

    keys = [c.key for c in machine.AUXILIARY_CIRCUITS if c.kind == "control"]
    assert len(keys) == 10, keys
    assert all(c.num_coils == 1 for c in machine.AUXILIARY_CIRCUITS if c.kind == "control")

    waveform = machine.control_waveform(480.0, 54.0, mode=2)
    upper = np.array([waveform[f"cc{m}u"] for m in range(1, 6)])
    # The pattern is pure n = 2 around the torus, and the two coils of a module agree.
    spectrum = np.abs(np.fft.rfft(upper) / len(upper)) * 2.0
    assert int(np.argmax(spectrum[1:])) + 1 == 2
    assert spectrum[2] == pytest.approx(480.0, rel=1e-9)
    assert spectrum[1] == pytest.approx(0.0, abs=1e-9)
    for module in range(1, 6):
        assert waveform[f"cc{module}u"] == pytest.approx(waveform[f"cc{module}l"])
    # Mode one on the same coils is the 1/1 pattern, which is a different harmonic.
    one = machine.control_waveform(480.0, 0.0, mode=1)
    single = np.abs(np.fft.rfft(np.array([one[f"cc{m}u"] for m in range(1, 6)])) / 5) * 2.0
    assert int(np.argmax(single[1:])) + 1 == 1


def test_every_module_attribute_an_analysis_names_exists():
    """A module attribute that does not exist is a NameError the run reaches at the end."""
    import ast
    import importlib
    import pkgutil

    import w7x_twin.analyses

    missing = []
    for info in pkgutil.walk_packages(
        w7x_twin.analyses.__path__, "w7x_twin.analyses."
    ):
        if info.ispkg:
            continue
        module = importlib.import_module(info.name)
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Names bound to an imported module, and names bound to anything else, which are
        # what tells a module attribute from an instance attribute.
        imported: dict[str, str] = {}
        assigned: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    target = f"{node.module}.{alias.name}"
                    try:
                        importlib.import_module(target)
                    except ImportError:
                        continue
                    imported[alias.asname or alias.name] = target
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    # ``import a.b`` binds ``a``, not ``a.b``; ``import a.b as c`` binds c.
                    if alias.asname:
                        imported[alias.asname] = alias.name
                    else:
                        imported[alias.name.split(".")[0]] = alias.name.split(".")[0]
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assigned.update(a.arg for a in node.args.args)

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
                continue
            name = node.value.id
            if name not in imported or name in assigned:
                continue
            target = importlib.import_module(imported[name])
            if hasattr(target, node.attr):
                continue
            # A submodule is an attribute only once it has been imported, so the name is
            # resolved as one before it is called missing.
            try:
                importlib.import_module(f"{imported[name]}.{node.attr}")
            except ImportError:
                missing.append(f"{info.name}:{node.lineno} {name}.{node.attr}")

    assert not missing, "attributes named on a module that does not carry them: " + ", ".join(
        missing
    )


def test_a_reversed_trace_measures_a_positive_arc():
    """A connection length is the two branches summed, so both have to be distances."""
    from w7x_twin.magnetics import fieldlines

    class Slab:
        """A uniform toroidal field, whose field lines are circles of known length."""

        num_field_periods = 1

        def __call__(self, r, phi, z):
            r = np.atleast_1d(np.asarray(r, dtype=float))
            zero = np.zeros_like(r)
            return zero, np.full_like(r, 2.5), zero

    radius = np.array([5.5])
    turns = 3
    lengths = []
    for sense in (+1, -1):
        rate = fieldlines._arc_rate(Slab(), radius, np.zeros(1), 0.0)
        dphi = sense * 2.0 * np.pi / 120
        lengths.append(float(np.sum(np.abs(rate * abs(dphi)))) * 120 * turns)
    assert lengths[0] > 0.0
    assert lengths[1] == pytest.approx(lengths[0])
    # A purely toroidal field's line is a circle, so its length is the circumference.
    assert lengths[0] == pytest.approx(turns * 2.0 * np.pi * 5.5, rel=1e-9)


def test_the_ring_measurement_recovers_a_known_pack_section():
    """A tube of rectangular ribs measures back to its own section."""
    from w7x_twin.hardware import cad

    width, height, ring_radius = 160.0, 223.4, 2000.0
    stations = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    corners = []
    for angle in stations:
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        vertical = np.array([0.0, 0.0, 1.0])
        centre = ring_radius * radial
        for a in (-0.5, 0.5):
            for b in (-0.5, 0.5):
                corners.append(centre + a * width * radial + b * height * vertical)
    corners = np.array(corners)
    # One station per bin is the exact regime: each bin holds one rib and nothing turns
    # inside it.
    w, h = cad.tube_sections(corners, corners, bins=120, minimum=4)
    assert len(w) >= 100
    assert float(np.median(np.minimum(w, h))) == pytest.approx(width, abs=0.01)
    assert float(np.median(np.maximum(w, h))) == pytest.approx(height, abs=0.01)
    # Coarser bins mix ribs the ring has turned between, which only widens the union, so
    # the binned figure is an upper bound and it is used as one.
    w, h = cad.tube_sections(corners, corners, bins=40)
    assert float(np.median(np.minimum(w, h))) >= width - 0.01
    assert float(np.median(np.minimum(w, h))) == pytest.approx(width, rel=0.02)
