import numpy as np

from EarthSystemGrids import GaussianLatLon, RotatedGaussianLatLon


def test_generate_mesh_area_and_shape_preserved():
    base_mesh = GaussianLatLon.generate_mesh(lat=8, lon=12)
    rotated_mesh = RotatedGaussianLatLon.generate_mesh(
        lat=8, lon=12, rotation_axis_longitude_deg=48.0, rotation_deg=12.0,
    )

    assert rotated_mesh.shape == base_mesh.shape
    np.testing.assert_allclose(rotated_mesh.area, base_mesh.area, atol=1e-3)
    np.testing.assert_array_equal(rotated_mesh.mask, np.ones_like(rotated_mesh.mask))
    # a non-trivial rotation should actually move the cell centres
    assert not np.allclose(rotated_mesh.face_lat, base_mesh.face_lat, atol=1e-6)


def test_generate_mesh_SCRIP_round_trip(tmp_path):
    from EarthSystemGrids.base import StructuredQuadMesh

    mesh = RotatedGaussianLatLon.generate_mesh(
        lat=8, lon=12, rotation_axis_longitude_deg=-42.0 + 90.0, rotation_deg=12.0,
        attrs={"title": "rotated test grid"},
    )

    angle = mesh.rotation_angle()
    assert np.all(np.isfinite(angle))

    scrip_file = tmp_path / "rotated_grid.nc"
    mesh.write_to_SCRIP_grid_file(scrip_file, flatten=True)
    loaded = StructuredQuadMesh.from_SCRIP_file(scrip_file)

    assert loaded.shape == mesh.shape
    np.testing.assert_allclose(loaded.face_lon, mesh.face_lon, atol=1e-8)
    np.testing.assert_allclose(loaded.face_lat, mesh.face_lat, atol=1e-8)
    np.testing.assert_allclose(loaded.area, mesh.area, atol=1e-3)
