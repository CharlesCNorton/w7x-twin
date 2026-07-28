"""Consolidated helpers against the implementations they replaced."""

from __future__ import annotations

import json

import numpy as np
import pytest

from w7x_twin.analyses import _common
from w7x_twin.hardware.walls import inside_contour
from w7x_twin.magnetics import fieldlines
from w7x_twin.plasma.kinetics import log_gradient


def reference_crossing_number(point_r, point_z, poly_r, poly_z):
    """The pre-consolidation crossing-number test, verbatim."""
    z0 = poly_z[:, None]
    z1 = np.roll(poly_z, -1)[:, None]
    r0 = poly_r[:, None]
    r1 = np.roll(poly_r, -1)[:, None]
    straddles = (z0 > point_z[None, :]) != (z1 > point_z[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = r0 + (point_z[None, :] - z0) * (r1 - r0) / (z1 - z0)
    hits = straddles & (point_r[None, :] < crossing)
    return (np.count_nonzero(hits, axis=0) % 2) == 1


def test_inside_contour_matches_reference_on_random_contours():
    rng = np.random.default_rng(11)
    for _ in range(20):
        theta = np.sort(rng.uniform(0.0, 2.0 * np.pi, rng.integers(8, 60)))
        radius = rng.uniform(0.5, 1.5, theta.size)
        poly_r = 5.5 + radius * np.cos(theta)
        poly_z = radius * np.sin(theta)
        point_r = rng.uniform(3.0, 8.0, 500)
        point_z = rng.uniform(-2.0, 2.0, 500)
        assert np.array_equal(
            inside_contour(point_r, point_z, poly_r, poly_z),
            reference_crossing_number(point_r, point_z, poly_r, poly_z),
        )


def test_inside_contour_on_the_unit_square():
    square_r = np.array([0.0, 1.0, 1.0, 0.0])
    square_z = np.array([0.0, 0.0, 1.0, 1.0])
    inside = inside_contour(
        np.array([0.5, 1.5, -0.1, 0.5]), np.array([0.5, 0.5, 0.5, 1.5]),
        square_r, square_z,
    )
    assert inside.tolist() == [True, False, False, False]


def test_log_gradient_matches_the_replaced_expression():
    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 5.0, 40)
    values[7] = 0.0
    radius = np.linspace(0.0, 0.5, 40)
    assert np.array_equal(
        log_gradient(values, radius),
        np.gradient(np.log(np.maximum(values, 1e-30)), radius),
    )


def test_strikes_concatenate_preserves_fields():
    def part(n, offset):
        return fieldlines.Strikes(
            struck=np.arange(n) % 2 == 0,
            r=np.arange(n, dtype=float) + offset,
            z=np.zeros(n),
            phi=np.full(n, 0.1 * offset),
            connection_length_m=np.full(n, float(offset)),
            start_r=np.arange(n, dtype=float),
            component=np.full(n, offset, dtype=int),
            component_names=["a", "b"],
        )

    merged = fieldlines.Strikes.concatenate([part(3, 1), part(2, 2)])
    assert merged.struck.tolist() == [True, False, True, True, False]
    assert merged.r.tolist() == [1.0, 2.0, 3.0, 2.0, 3.0]
    assert merged.component.tolist() == [1, 1, 1, 2, 2]
    assert merged.component_names == ["a", "b"]


def test_midplane_island_span_measures_the_widest_line():
    section = fieldlines.Poincare(
        r=np.array([6.0, 6.05, 6.2, 6.21, 5.0, 6.4]),
        z=np.array([0.0, 0.01, 0.0, -0.01, 0.0, 0.5]),
        line_index=np.array([0, 0, 1, 1, 2, 3]),
        plane_phi=0.0,
        turns_completed=np.zeros(6, dtype=int),
    )
    width, lines = fieldlines.midplane_island_span(
        section, r_axis=5.5, z_axis=0.0, min_points=2
    )
    assert lines == 2
    assert abs(width - 0.05) < 1e-12
    none, count = fieldlines.midplane_island_span(section, 7.0, 0.0)
    assert np.isnan(none) and count == 0


def test_write_record_stamps_geometry_and_encodes_numpy(tmp_path):
    class Geometry:
        def as_dict(self):
            return {"geometry": "abc"}

    path = _common.write_record(
        tmp_path / "record.json",
        {"value": np.float64(1.5), "array": np.arange(3), "flag": np.bool_(True)},
        geometry=Geometry(),
    )
    stored = json.loads(path.read_text())
    assert list(stored) == ["geometry", "value", "array", "flag"]
    assert stored["array"] == [0, 1, 2] and stored["value"] == 1.5


def test_write_record_stamps_input_digests(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('{"value": 1}')
    record = _common.write_record(
        tmp_path / "record.json", {"answer": 2}, reads=(source,)
    )
    stored = json.loads(record.read_text())
    assert stored["reads"] == {str(source): _common.file_digest(source)}
    assert stored["reads"][str(source)] != "absent"
    assert _common.file_digest(tmp_path / "missing.json") == "absent"


def test_dependency_check_flags_a_mutated_input(tmp_path, capsys):
    from w7x_twin.analyses import data

    source = tmp_path / "input.json"
    source.write_text('{"value": 1}')
    _common.write_record(tmp_path / "record.json", {"answer": 2}, reads=(source,))
    data._records.clear()
    data.dependency_checks(tmp_path)
    assert [record["agrees"] for record in data._records] == [True]

    source.write_text('{"value": 2}')
    data._records.clear()
    data.dependency_checks(tmp_path)
    assert [record["agrees"] for record in data._records] == [False]
    data._records.clear()
    capsys.readouterr()


def test_table_header_and_rows_share_one_layout(capsys):
    table = _common.Table(("name", "6s"), ("value", "7.3f"), ("note", "s"))
    table.begin()
    table.row("a", 1.25, "-")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"{'name':<6s} {'value':>7s} note"
    assert out[1] == "-" * len(out[0])
    assert out[2] == f"{'a':6s} {1.25:7.3f} -"
    assert out[0].index("value") + len("value") == out[2].index("1.250") + len("1.250")


def _toroidal_field(field_t: float = 2.5, r0: float = 5.5):
    """A pure 1/R toroidal field on a grid, as a VacuumField without a response table."""
    from w7x_twin.magnetics.field import VacuumField

    vacuum = object.__new__(VacuumField)
    vacuum.num_r, vacuum.num_z, vacuum.num_phi = 121, 121, 36
    vacuum.r_min, vacuum.r_max = 4.3, 6.7
    vacuum.z_min, vacuum.z_max = -1.2, 1.2
    vacuum.num_field_periods = 5
    vacuum.period = 2.0 * np.pi / 5
    vacuum.dr = (vacuum.r_max - vacuum.r_min) / (vacuum.num_r - 1)
    vacuum.dz = (vacuum.z_max - vacuum.z_min) / (vacuum.num_z - 1)
    vacuum.dphi = vacuum.period / vacuum.num_phi
    radius = vacuum.r_min + vacuum.dr * np.arange(vacuum.num_r)
    vacuum.b = np.zeros((3, vacuum.num_phi, vacuum.num_z, vacuum.num_r))
    vacuum.b[1] = (field_t * r0 / radius)[None, None, :]
    vacuum._digest = None
    return vacuum


def test_field_gradient_matches_the_interpolant():
    vacuum = _toroidal_field()
    rng = np.random.default_rng(5)
    r = rng.uniform(4.5, 6.5, 200)
    z = rng.uniform(-1.0, 1.0, 200)
    phi = rng.uniform(0.0, 2.0 * np.pi, 200)
    _, gradient = vacuum.with_gradient(r, phi, z)
    h = 1e-6
    for axis, offset in ((0, (h, 0.0, 0.0)), (2, (0.0, 0.0, h))):
        upper = np.stack(vacuum(r + offset[0], phi, z + offset[2]))
        lower = np.stack(vacuum(r - offset[0], phi, z - offset[2]))
        numerical = (upper - lower) / (2.0 * h)
        assert np.nanmax(np.abs(gradient[:, axis] - numerical)) < 1e-4


def test_guiding_centre_reproduces_the_toroidal_drift():
    from w7x_twin.plasma import transport

    vacuum = _toroidal_field(field_t=2.5, r0=5.5)
    energy_ev = 55.0e3
    mass = transport.PROTON_MASS
    speed = np.sqrt(2.0 * transport.ELEMENTARY_CHARGE * energy_ev / mass)
    pitch = 0.6
    radius = np.array([5.5])
    phi = np.array([0.0])
    height = np.array([0.0])
    v_par = np.array([speed * pitch])
    field_t = 2.5
    mu_over_m = np.array([0.5 * speed**2 * (1.0 - pitch**2) / field_t])
    charge_over_m = transport.ELEMENTARY_CHARGE / mass

    _, dphi_dt, dz_dt, dv_dt = transport.guiding_centre_rates(
        vacuum, radius, phi, height, v_par, mu_over_m, charge_over_m
    )
    # In B = B0 R0 / R the grad-B and curvature drifts are both vertical and upward
    # for a positive ion: v_z = (v_par^2 + v_perp^2 / 2) / ((q/m) B R).
    analytic = (v_par[0] ** 2 + 0.5 * speed**2 * (1.0 - pitch**2)) / (
        charge_over_m * field_t * 5.5
    )
    assert abs(dz_dt[0] - analytic) < 5e-3 * abs(analytic)
    assert abs(dv_dt[0]) < 1e-3 * speed
    assert abs(dphi_dt[0] * 5.5 - v_par[0]) < 1e-3 * speed


def _synthetic_tables(viscous: float) -> "object":
    """Monoenergetic coefficients with a set viscous reduction of D33."""
    from w7x_twin.plasma.neoclassical import MonoenergeticCoefficients

    nu = np.logspace(-6, -1, 12)
    return MonoenergeticCoefficients(
        s=0.2,
        collisionality=nu,
        radial_field=np.zeros_like(nu),
        d11=1e-3 * nu ** -0.5,
        d31=np.full_like(nu, -0.5),
        d33=np.full_like(nu, 1.0),
        d33_spitzer=np.full_like(nu, 1.0 + viscous),
    )


def test_restored_flow_friction_reproduces_the_spitzer_factor():
    from w7x_twin.plasma import neoclassical

    for charge, factor in zip(
        neoclassical.SPITZER_CHARGES[:-1], neoclassical.SPITZER_FACTORS[:-1]
    ):
        l11 = charge
        l12 = 1.5 * charge
        l22 = l12 * l12 / (l11 * (1.0 - factor))
        assert abs((l11 - l12 * l12 / l22) / l11 - factor) < 1e-12


def test_channel_correction_limits():
    from w7x_twin.plasma import neoclassical

    keywords = dict(
        density_m3=8.0e19, electron_temperature_ev=2000.0,
        ion_temperature_ev=1100.0, density_gradient=-2.0,
        electron_temperature_gradient=-4.0, ion_temperature_gradient=-4.0,
    )
    ordinary = neoclassical.channel_correction(
        _synthetic_tables(viscous=1.0), z_effective=1.0, **keywords
    )
    assert np.isfinite(ordinary["relative_correction"])
    assert abs(ordinary["relative_correction"]) < 0.1

    # With no viscosity the conserving friction has an exact Galilean zero mode; the
    # minimum-norm gauge must return finite flows rather than fall back or blow up.
    free = neoclassical.restored_flows(
        _synthetic_tables(viscous=0.0), z_effective=1.0, **keywords
    )
    assert np.isfinite(free["electron_flow"]) and np.isfinite(free["ion_flow"])

    # The correction is independent of the tabulated D31 sign convention: the kernel
    # and the drives flip together.
    flipped_tables = _synthetic_tables(viscous=1.0)
    flipped = neoclassical.channel_correction(
        neoclassical.MonoenergeticCoefficients(
            s=flipped_tables.s,
            collisionality=flipped_tables.collisionality,
            radial_field=flipped_tables.radial_field,
            d11=flipped_tables.d11,
            d31=-flipped_tables.d31,
            d33=flipped_tables.d33,
            d33_spitzer=flipped_tables.d33_spitzer,
        ),
        z_effective=1.0, **keywords,
    )
    assert flipped["delta_q_over_nt"] == pytest.approx(
        ordinary["delta_q_over_nt"], rel=1e-9
    )


def test_axis_memo_round_trips_through_the_cache_file(tmp_path):
    fieldlines._AXIS_MEMO.clear()

    class Uniform:
        num_field_periods = 5

        def digest(self):
            return "uniformfield"

        def __call__(self, r, phi, z):
            shape = np.shape(np.atleast_1d(r))
            return np.zeros(shape), np.ones(shape), np.zeros(shape)

    first = fieldlines.find_axis(
        Uniform(), r_guess=5.9, iterations=2, cache_dir=tmp_path
    )
    stored = json.loads((tmp_path / "axes.json").read_text())
    assert list(stored.values()) == [list(first)]
    fieldlines._AXIS_MEMO.clear()
    again = fieldlines.find_axis(
        Uniform(), r_guess=5.9, iterations=2, cache_dir=tmp_path
    )
    assert again == first
