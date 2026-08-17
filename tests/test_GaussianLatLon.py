import numpy as np

from EarthSystemGrids import GaussianLatLon
from EarthSystemGrids.base import StructuredQuadMesh, _R_EARTH

_LAT_BOUNDS = GaussianLatLon.gaussian_latitude_bounds(8)
_LON_BOUNDS = GaussianLatLon.equally_spaced_bounds(12, 0.0, 360.0)


def test_generate_mesh_area_sums_to_sphere():
    mesh = GaussianLatLon.generate_mesh(lat=_LAT_BOUNDS, lon=_LON_BOUNDS)

    assert mesh.shape == (8, 12)
    np.testing.assert_allclose(mesh.area.sum(), 4 * np.pi * _R_EARTH**2, rtol=1e-6)
    np.testing.assert_array_equal(mesh.mask, np.ones_like(mesh.mask))


def test_generate_mesh_SCRIP_round_trip(tmp_path):
    mesh = GaussianLatLon.generate_mesh(
        lat=_LAT_BOUNDS, lon=_LON_BOUNDS, attrs={"title": "Gaussian lat-lon test grid"},
    )

    scrip_file = tmp_path / "gaussian_grid.nc"
    mesh.write_to_SCRIP_grid_file(scrip_file, flatten=True)
    loaded = StructuredQuadMesh.from_SCRIP_file(scrip_file)

    assert loaded.shape == mesh.shape
    np.testing.assert_allclose(loaded.face_lon, mesh.face_lon, atol=1e-8)
    np.testing.assert_allclose(loaded.face_lat, mesh.face_lat, atol=1e-8)
    np.testing.assert_allclose(loaded.area, mesh.area, rtol=1e-6)
    np.testing.assert_array_equal(loaded.mask, mesh.mask)
    assert loaded.attrs.get("title") == "Gaussian lat-lon test grid"


def test_generate_mesh_CF_round_trip(tmp_path):
    mesh = GaussianLatLon.generate_mesh(
        lat=_LAT_BOUNDS, lon=_LON_BOUNDS, attrs={"title": "Gaussian lat-lon test grid"},
    )

    cf_file = tmp_path / "gaussian_grid_cf.nc"
    mesh.write_to_CF_grid_file(cf_file)
    loaded = StructuredQuadMesh.from_CF_file(cf_file)

    assert loaded.shape == mesh.shape
    np.testing.assert_allclose(loaded.face_lon, mesh.face_lon, atol=1e-8)
    np.testing.assert_allclose(loaded.face_lat, mesh.face_lat, atol=1e-8)
    np.testing.assert_allclose(loaded.area, mesh.area, rtol=1e-6)
    np.testing.assert_array_equal(loaded.mask, mesh.mask)
    assert loaded.attrs.get("title") == "Gaussian lat-lon test grid"
