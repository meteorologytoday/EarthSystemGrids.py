import numpy as np
import xarray as xr

from EarthSystemGrids.base.StructuredQuadMesh import StructuredQuadMesh
from EarthSystemGrids.base.UnstructuredGridMesh import UnstructuredGridMesh


def test_from_SCRIP_file_generic_reads_structured_grid(tmp_path):
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

    mesh = StructuredQuadMesh.from_corners(
        corner_lon, corner_lat, face_lon, face_lat, area, mask, (nj, ni),
    )
    scrip_file = tmp_path / "structured.nc"
    mesh.write_to_SCRIP_grid_file(scrip_file, flatten=True)

    # Read the same file with the generic, floating-point-matching reader
    # instead of StructuredQuadMesh's structured-index one -- the mesh
    # comes back as an unordered 1D collection of faces (shape == (nface,)),
    # so compare by sorting rather than by position.
    loaded = UnstructuredGridMesh.from_SCRIP_file(scrip_file)

    assert loaded.shape == (nj * ni,)
    np.testing.assert_allclose(np.sort(loaded.face_lon), np.sort(mesh.face_lon), atol=1e-10)
    np.testing.assert_allclose(np.sort(loaded.face_lat), np.sort(mesh.face_lat), atol=1e-10)
    np.testing.assert_allclose(np.sort(loaded.area), np.sort(mesh.area), atol=1e-6)
    np.testing.assert_array_equal(np.sort(loaded.mask), np.sort(mesh.mask))
    # every shared corner should have been merged onto one node
    assert len(loaded.node_lon) == (nj + 1) * (ni + 1)


def test_from_SCRIP_file_trims_padded_triangle_corners(tmp_path):
    # A quad and a triangle, the latter padded to grid_corners=4 by
    # repeating its last corner -- the SCRIP convention for mixed-arity
    # meshes (e.g. MPAS output).
    ds = xr.Dataset(
        data_vars=dict(
            grid_dims=(["grid_rank"], [2]),
            grid_imask=(["grid_size"], [1, 1]),
            grid_center_lat=(["grid_size"], [0.5, 10.0]),
            grid_center_lon=(["grid_size"], [0.5, 10.0]),
            grid_corner_lat=(["grid_size", "grid_corners"], [[0, 0, 1, 1], [10, 11, 11, 11]]),
            grid_corner_lon=(["grid_size", "grid_corners"], [[0, 1, 1, 0], [10, 10, 11, 11]]),
            grid_area=(["grid_size"], [1e-4, 1e-4]),
        ),
    )
    for name in ("grid_corner_lat", "grid_corner_lon", "grid_center_lat", "grid_center_lon"):
        ds[name].attrs["units"] = "degrees"
    scrip_file = tmp_path / "mixed_arity.nc"
    ds.to_netcdf(scrip_file)

    mesh = UnstructuredGridMesh.from_SCRIP_file(scrip_file)

    assert mesh.shape == (2,)
    np.testing.assert_array_equal(mesh.n_nodes_per_face, [4, 3])
