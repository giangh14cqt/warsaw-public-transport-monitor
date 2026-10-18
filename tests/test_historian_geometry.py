"""
Unit tests for Warsaw metric projection and linear referencing geometry.
"""

import math

from src.historian.matching.geometry import (
    WarsawShapePolyline,
    haversine_distance,
    interpolate_stop_boundary_timestamp,
    inverse_warsaw_metric,
    metric_distance,
    project_to_warsaw_metric,
)


def test_metric_projection_roundtrip():
    lat, lon = 52.2297, 21.0122  # Warsaw city center
    x, y = project_to_warsaw_metric(lat, lon)
    inv_lat, inv_lon = inverse_warsaw_metric(x, y)

    assert math.isclose(lat, inv_lat, abs_tol=1e-6)
    assert math.isclose(lon, inv_lon, abs_tol=1e-6)


def test_metric_vs_haversine_accuracy():
    # 1 km displacement in Warsaw
    lat1, lon1 = 52.2300, 21.0000
    lat2, lon2 = 52.2300, 21.01465

    d_metric = metric_distance(lat1, lon1, lat2, lon2)
    d_haversine = haversine_distance(lat1, lon1, lat2, lon2)

    # Error against spherical haversine must be under 0.5% (< 5 meters per kilometer)
    assert abs(d_metric - d_haversine) / d_haversine < 0.005


def test_interpolate_stop_boundary_timestamp():
    # Vehicle approaches stop zone:
    # At t=100s, d=35m (outside 25m)
    # At t=110s, d=15m (inside 25m)
    # Crossing at 25m occurs exactly halfway: t=105s
    t_outside = 100.0
    d_outside = 35.0
    t_inside = 110.0
    d_inside = 15.0

    t_entry = interpolate_stop_boundary_timestamp(
        t_outside, d_outside, t_inside, d_inside, radius_meters=25.0
    )
    assert math.isclose(t_entry, 105.0, abs_tol=1e-3)


def test_warsaw_shape_polyline_projection():
    # Straight segment moving East from (52.23, 21.00) to (52.23, 21.01)
    # Then turning North to (52.24, 21.01)
    pts = [(52.2300, 21.0000), (52.2300, 21.0100), (52.2400, 21.0100)]
    poly = WarsawShapePolyline(pts)

    assert poly.total_length > 1000.0

    # Test point halfway along first segment
    along, cross, seg_idx = poly.project_point(52.2300, 21.0050)
    assert math.isclose(cross, 0.0, abs_tol=1e-1)
    assert seg_idx == 0
    assert 300.0 < along < 400.0
