import numpy as np

from EarthSystemGrids.base.sphere_rotation import rotate_lonlat


def test_rotate_lonlat_moves_pole_as_expected():
    # True pole (lon=0, lat=90deg) rotated 90deg (right-hand rule) about
    # an axis at longitude 0 (the point [1, 0, 0]) lands on [0, -1, 0],
    # i.e. lon=-90deg, lat=0deg -- hand-computable via Rodrigues by
    # inspection: k x p = (1,0,0) x (0,0,1) = (0,-1,0).
    lon = np.array([0.0])
    lat = np.array([np.pi / 2])

    lon_rot, lat_rot = rotate_lonlat(lon, lat, axis_longitude_deg=0.0, rotation_deg=90.0)

    np.testing.assert_allclose(lon_rot, [-np.pi / 2], atol=1e-10)
    np.testing.assert_allclose(lat_rot, [0.0], atol=1e-10)


def test_rotate_lonlat_zero_rotation_is_identity():
    lon = np.deg2rad(np.array([0.0, 30.0, 190.0, -170.0]))
    lat = np.deg2rad(np.array([0.0, 45.0, -60.0, 89.0]))

    lon_rot, lat_rot = rotate_lonlat(lon, lat, axis_longitude_deg=123.0, rotation_deg=0.0)

    np.testing.assert_allclose(np.cos(lon_rot), np.cos(lon), atol=1e-10)
    np.testing.assert_allclose(np.sin(lon_rot), np.sin(lon), atol=1e-10)
    np.testing.assert_allclose(lat_rot, lat, atol=1e-10)


def test_rotate_lonlat_preserves_shape():
    lon = np.zeros((4, 6, 4))
    lat = np.zeros((4, 6, 4))
    lon_rot, lat_rot = rotate_lonlat(lon, lat, axis_longitude_deg=30.0, rotation_deg=15.0)
    assert lon_rot.shape == (4, 6, 4)
    assert lat_rot.shape == (4, 6, 4)
