import numpy as np

from EarthSystemGrids.base import StructuredQuadMesh
from EarthSystemGrids.DisplacedPoleGrid import build_example_grid


def _build_test_mesh():
    # Small enough to build fast -- the SCRIP/CF I/O contract under test
    # doesn't depend on grid resolution. The default mi1996 formulation logs
    # an internal parameter-feasibility warning at this resolution (about
    # the f/g rate schedule); that's unrelated to the mesh geometry or the
    # round trip being tested here.
    grid = build_example_grid(number_of_rows_in_NH=4, number_of_columns=8, dlat_in_SH_degree=30.0)
    return grid.generate_mesh()


def test_generate_mesh_shape_and_area():
    mesh = _build_test_mesh()

    assert len(mesh.shape) == 2
    assert np.all(mesh.area > 0)
    np.testing.assert_array_equal(mesh.mask, np.ones_like(mesh.mask))


def test_generate_mesh_SCRIP_round_trip(tmp_path):
    mesh = _build_test_mesh()

    scrip_file = tmp_path / "displaced_pole_grid.nc"
    mesh.write_to_SCRIP_grid_file(scrip_file, flatten=True)
    loaded = StructuredQuadMesh.from_SCRIP_file(scrip_file)

    assert loaded.shape == mesh.shape
    np.testing.assert_allclose(loaded.face_lon, mesh.face_lon, atol=1e-6)
    np.testing.assert_allclose(loaded.face_lat, mesh.face_lat, atol=1e-6)
    np.testing.assert_allclose(loaded.area, mesh.area, rtol=1e-6)
    np.testing.assert_array_equal(loaded.mask, mesh.mask)
    assert loaded.attrs.get("title") == mesh.attrs["title"]


def test_generate_mesh_CF_round_trip(tmp_path):
    mesh = _build_test_mesh()

    cf_file = tmp_path / "displaced_pole_grid_cf.nc"
    mesh.write_to_CF_grid_file(cf_file)
    loaded = StructuredQuadMesh.from_CF_file(cf_file)

    assert loaded.shape == mesh.shape
    np.testing.assert_allclose(loaded.face_lon, mesh.face_lon, atol=1e-6)
    np.testing.assert_allclose(loaded.face_lat, mesh.face_lat, atol=1e-6)
    np.testing.assert_allclose(loaded.area, mesh.area, rtol=1e-6)
    np.testing.assert_array_equal(loaded.mask, mesh.mask)
    assert loaded.attrs.get("title") == mesh.attrs["title"]
