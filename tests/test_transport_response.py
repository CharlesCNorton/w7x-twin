"""The measured saturation response, its consumers, and the placements built on it."""

import json
import types

import numpy as np
import pytest

from w7x_twin.plasma import kinetics, transport


def response_record(path, points):
    path.write_text(json.dumps({"points": points}))
    return path


def point(s, gradient, x, q, electron=None, box=(24, 16)):
    out = {
        "torflux": s,
        "gradient": gradient,
        "mixing_length_sum": x,
        "saturated_ion_heat_flux_gyrobohm": q,
        "box": list(box),
    }
    if electron is not None:
        out["saturated_electron_heat_flux_gyrobohm"] = electron
    return out


def test_the_largest_box_supersedes_the_grid_point(tmp_path):
    record = response_record(
        tmp_path / "constant.json",
        [
            point(0.25, 4.5, 1.0, 500.0),
            point(0.25, 4.5, 1.0, 160.0, box=(48, 32)),
            point(0.25, 1.5, 0.3, 0.0),
        ],
    )
    response = transport.MixingLengthResponse.read(record)
    assert response(0.25, 1.0) == pytest.approx(160.0)
    assert response(0.25, 0.3) == pytest.approx(0.0)


def test_the_electron_curve_reads_its_own_channel_and_falls_back(tmp_path):
    with_electrons = transport.MixingLengthResponse.read(
        response_record(
            tmp_path / "with.json",
            [
                point(0.5, 1.5, 0.4, 1.0, electron=0.5),
                point(0.5, 4.5, 1.2, 9.0, electron=4.0),
            ],
        )
    )
    assert with_electrons(0.5, 1.2, species="electron") == pytest.approx(4.0)
    assert with_electrons(0.5, 1.2, species="ion") == pytest.approx(9.0)

    without = transport.MixingLengthResponse.read(
        response_record(
            tmp_path / "without.json",
            [point(0.5, 1.5, 0.4, 1.0), point(0.5, 4.5, 1.2, 9.0)],
        )
    )
    assert without(0.5, 1.2, species="electron") == pytest.approx(9.0)


def test_the_local_closure_divides_the_flux_by_its_gradient(tmp_path):
    cases = [
        {
            "configuration": "standard",
            "torflux": s,
            "tprim": t,
            "fprim": 1.0,
            "ky": 1.0,
            "growth_rate": 0.5 * t,
        }
        for s in (0.2, 0.8)
        for t in (1.5, 4.5)
    ]
    table = transport.GrowthRateTable.from_cases(cases)
    assert table.mixing_length_sum(0.2, 4.5, 1.0) == pytest.approx(2.25)

    response = transport.MixingLengthResponse.read(
        response_record(
            tmp_path / "constant.json",
            [
                point(0.2, 1.5, 0.75, 0.0),
                point(0.2, 4.5, 2.25, 90.0),
                point(0.8, 1.5, 0.75, 0.0),
                point(0.8, 4.5, 2.25, 90.0),
            ],
        )
    )
    chi = transport.local_turbulence(table, response, field_t=2.5, minor_radius_m=0.5)
    unit = transport.gyro_bohm(0.55 * 1000.0, 2.5, 0.5)
    # At a calibration point the closure returns exactly flux over gradient.
    assert chi(0.2, 4.5, 1.0, 1000.0, 8.0e19) == pytest.approx(90.0 / 4.5 * unit)


def _stub_output(ns=31):
    wout = types.SimpleNamespace(
        ns=ns,
        vp=np.ones(ns),
        volume_p=30.0,
        Aminor_p=0.5,
        Rmajor_p=5.5,
        b0=2.5,
        iotaf=np.linspace(0.85, 0.95, ns),
        phi=np.linspace(0.0, 2.0, ns),
    )
    return types.SimpleNamespace(wout=wout)


def test_the_march_settles_on_the_cliff_instead_of_falling_off_it():
    def cliff(s, a_lt, a_ln, temperature_ev, density):
        return 0.0 if a_lt < 2.0 else 50.0 * (a_lt - 2.0)

    def channels(s, electron_t, ion_t, n):
        return np.full_like(s, 0.05), np.full_like(s, 0.05)

    solution = transport.solve_split(
        _stub_output(),
        kinetics.KineticProfiles(),
        transport.Heating(power_w=4.0e6),
        channels,
        turbulent_local=cliff,
        chi_updates=2,
        inner_iterations=25,
    )
    electron = solution.electron_temperature_ev
    assert electron[0] > 3.0 * electron[-1]
    assert np.all(np.diff(electron) < 1.0)
    assert np.isfinite(electron).all()


def test_noble_interfaces_avoid_low_order_rationals():
    from w7x_twin.analyses.equilibrium import noble_interfaces

    grid = np.linspace(0.0, 1.0, 401)
    iota = 0.81 + 0.10 * grid
    resonance = 5.0 / 6.0
    placed = noble_interfaces(8, grid, iota, resonance)
    assert len(placed) == 7
    iotas = np.interp(placed, grid, iota)
    assert (iotas < resonance).any() and (iotas > resonance).any()
    for value in iotas:
        for q in range(2, 25):
            for p in range(int(0.80 * q), int(0.92 * q) + 2):
                assert abs(value - p / q) >= 0.25 / q**2 - 1e-9
    assert np.min(np.diff(np.sort(iotas))) >= 0.5 * (iota[-1] - iota[0]) / 8 - 1e-6


def test_a_pinned_interface_sits_on_the_resonance():
    from w7x_twin.analyses.equilibrium import noble_interfaces

    grid = np.linspace(0.0, 1.0, 401)
    iota = 0.81 + 0.10 * grid
    resonance = 5.0 / 6.0
    placed = noble_interfaces(6, grid, iota, resonance, pinned=True)
    iotas = np.interp(placed, grid, iota)
    assert np.min(np.abs(iotas - resonance)) == pytest.approx(0.0, abs=1e-6)


def test_the_shear_quench_suppresses_only_sheared_surfaces():
    cases = [
        {
            "configuration": "standard",
            "torflux": s,
            "tprim": t,
            "fprim": 1.0,
            "ky": 1.0,
            "growth_rate": 0.2,
        }
        for s in (0.2, 0.8)
        for t in (1.5, 4.5)
    ]
    table = transport.GrowthRateTable.from_cases(cases)
    quench = transport.shear_quench_model(table, field_t=2.5, minor_radius_m=0.5)

    s = np.linspace(0.05, 0.95, 20)
    temperature = 2000.0 * (1.0 - s) + 100.0
    density = np.full_like(s, 8.0e19)

    calm = quench(s, np.zeros_like(s), temperature, temperature, density)
    assert np.allclose(calm, 1.0)

    sheared = quench(
        s, 4.0e5 * np.sqrt(s), temperature, temperature, density
    )
    assert np.all(sheared <= 1.0)
    assert np.any(sheared < 0.5)
