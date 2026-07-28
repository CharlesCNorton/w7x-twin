"""Consolidated helpers against the implementations they replaced."""

from __future__ import annotations

import json

import numpy as np

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


def test_table_header_and_rows_share_one_layout(capsys):
    table = _common.Table(("name", "6s"), ("value", "7.3f"), ("note", "s"))
    table.begin()
    table.row("a", 1.25, "-")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"{'name':<6s} {'value':>7s} note"
    assert out[1] == "-" * len(out[0])
    assert out[2] == f"{'a':6s} {1.25:7.3f} -"
    assert out[0].index("value") + len("value") == out[2].index("1.250") + len("1.250")


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
