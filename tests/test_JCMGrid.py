import numpy as np

from EarthSystemGrids.base import StructuredQuadMesh, _R_EARTH
from EarthSystemGrids.JCMGrid import generate_JCMGrid

# T21 is the coarsest valid truncation (see jcm.utils.VALID_TRUNCATIONS),
# which keeps the grid small and the test fast.
_RESOLUTION = 21


def test_generate_JCMGrid_area_sums_to_sphere():
    mesh = generate_JCMGrid(_RESOLUTION)

    assert len(mesh.shape) == 2
    np.testing.assert_allclose(mesh.area.sum(), 4 * np.pi * _R_EARTH**2, rtol=1e-6)
    np.testing.assert_array_equal(mesh.mask, np.ones_like(mesh.mask))


def test_generate_JCMGrid_SCRIP_round_trip(tmp_path):
    mesh = generate_JCMGrid(_RESOLUTION, attrs={"title": "JCM test grid"})

    scrip_file = tmp_path / "jcm_grid.nc"
    mesh.write_to_SCRIP_grid_file(scrip_file, flatten=True)
    loaded = StructuredQuadMesh.from_SCRIP_file(scrip_file)

    assert loaded.shape == mesh.shape
    np.testing.assert_allclose(loaded.face_lon, mesh.face_lon, atol=1e-8)
    np.testing.assert_allclose(loaded.face_lat, mesh.face_lat, atol=1e-8)
    np.testing.assert_allclose(loaded.area, mesh.area, rtol=1e-6)
    np.testing.assert_array_equal(loaded.mask, mesh.mask)
    assert loaded.attrs.get("title") == "JCM test grid"


def test_generate_JCMGrid_CF_round_trip(tmp_path):
    mesh = generate_JCMGrid(_RESOLUTION, attrs={"title": "JCM test grid"})

    cf_file = tmp_path / "jcm_grid_cf.nc"
    mesh.write_to_CF_grid_file(cf_file)
    loaded = StructuredQuadMesh.from_CF_file(cf_file)

    assert loaded.shape == mesh.shape
    np.testing.assert_allclose(loaded.face_lon, mesh.face_lon, atol=1e-8)
    np.testing.assert_allclose(loaded.face_lat, mesh.face_lat, atol=1e-8)
    np.testing.assert_allclose(loaded.area, mesh.area, rtol=1e-6)
    np.testing.assert_array_equal(loaded.mask, mesh.mask)
    assert loaded.attrs.get("title") == "JCM test grid"
