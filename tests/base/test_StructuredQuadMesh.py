import numpy as np
import pytest

from EarthSystemGrids.base.StructuredQuadMesh import StructuredQuadMesh


def _build_test_mesh():
    nj, ni = 4, 6
    lon_edges = np.linspace(0, 2 * np.pi, ni + 1)
    lat_edges = np.linspace(-np.pi / 3, np.pi / 3, nj + 1)

    corner_lon = np.empty((nj, ni, 4))
    corner_lat = np.empty((nj, ni, 4))
    for j in range(nj):
        for i in range(ni):
            corner_lon[j, i] = [lon_edges[i], lon_edges[i + 1], lon_edges[i + 1], lon_edges[i]]
            corner_lat[j, i] = [lat_edges[j], lat_edges[j], lat_edges[j + 1], lat_edges[j + 1]]

    face_lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])[None, :].repeat(nj, axis=0)
    face_lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])[:, None].repeat(ni, axis=1)
    area = np.ones((nj, ni))
    mask = np.ones((nj, ni), dtype=int)
    mask[0, 0] = 0

    return StructuredQuadMesh.from_corners(
        corner_lon, corner_lat, face_lon, face_lat, area, mask, (nj, ni),
        attrs={"title": "test grid"},
    )


@pytest.mark.parametrize("flatten", [True, False])
def test_from_SCRIP_file_round_trip(tmp_path, flatten):
    mesh = _build_test_mesh()
    scrip_file = tmp_path / f"grid_flatten-{flatten}.nc"
    mesh.write_to_SCRIP_grid_file(scrip_file, flatten=flatten)

    loaded = StructuredQuadMesh.from_SCRIP_file(scrip_file)

    assert loaded.shape == mesh.shape
    np.testing.assert_allclose(loaded.face_lon, mesh.face_lon, atol=1e-10)
    np.testing.assert_allclose(loaded.face_lat, mesh.face_lat, atol=1e-10)
    np.testing.assert_allclose(loaded.area, mesh.area, atol=1e-6)
    np.testing.assert_array_equal(loaded.mask, mesh.mask)
    np.testing.assert_allclose(loaded.node_lon, mesh.node_lon, atol=1e-8)
    np.testing.assert_allclose(loaded.node_lat, mesh.node_lat, atol=1e-8)
    np.testing.assert_allclose(loaded.rotation_angle(), mesh.rotation_angle(), atol=1e-8)
    assert loaded.attrs.get("title") == "test grid"
