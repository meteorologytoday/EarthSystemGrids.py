from datetime import datetime, timezone

import numpy as np
import xarray as xr


_R_EARTH = 6.371e6  # m, spherical Earth


class GridMesh:
    """
    UGRID-native storage for a structured mesh with uniform N-sided cells.

    All angular quantities are in radians. All linear quantities are in metres.
    The 1D storage is the canonical form; use .shape to recover the logical
    M-dimensional arrangement.

    Attributes
    ----------
    node_lon, node_lat : (nnode,)        unique corner coordinates, radians
    edge_nodes         : (nedge, 2) int  node indices of each edge's endpoints
    edge_length        : (nedge,)        great-circle length, metres
    face_nodes         : (nface, N) int  node indices per face, CCW
    face_edges         : (nface, N) int  edge indices per face;
                                         edge k connects face_nodes[k] to
                                         face_nodes[(k+1) % N]
    face_lon, face_lat : (nface,)        cell-centre coordinates, radians
    area               : (nface,)        cell area, m²
    mask               : (nface,) int    0 = land, 1 = ocean
    shape              : tuple           logical arrangement, e.g. (nj, ni)
    n_corners          : int             N, uniform for all faces
    attrs              : dict            grid-specific metadata
    """

    def __init__(
        self,
        node_lon,
        node_lat,
        edge_nodes,
        edge_length,
        face_nodes,
        face_edges,
        face_lon,
        face_lat,
        area,
        mask,
        shape,
        n_corners,
        attrs=None,
    ):
        self.node_lon    = node_lon
        self.node_lat    = node_lat
        self.edge_nodes  = edge_nodes
        self.edge_length = edge_length
        self.face_nodes  = face_nodes
        self.face_edges  = face_edges
        self.face_lon    = face_lon
        self.face_lat    = face_lat
        self.area        = area
        self.mask        = mask
        self.shape       = shape
        self.n_corners   = n_corners
        self.attrs       = attrs or {}

    @classmethod
    def from_corners(cls, corner_lon, corner_lat, face_lon, face_lat,
                     area, mask, shape, n_corners=4, attrs=None):
        """
        Build a GridMesh from per-face corner coordinate arrays.

        Parameters
        ----------
        corner_lon, corner_lat : (*shape, n_corners)  radians, CCW from SW corner
        face_lon,   face_lat   : (*shape,)             radians, cell centres
        area                   : (*shape,)             m², cell area
        mask                   : (*shape,)             int, 0=land 1=ocean
        shape                  : logical arrangement, e.g. (nj, ni)
        n_corners              : int, must be 4 for 2-D shape
        attrs                  : dict of grid-specific metadata
        """
        if len(shape) == 2:
            node_lon, node_lat, face_nodes, edge_nodes, face_edges = \
                _build_topology_2d(corner_lon, corner_lat, shape, n_corners)
        else:
            raise NotImplementedError(
                f"from_corners only supports 2D shape; got {len(shape)}D. "
                "CubeSphere (3D) requires tile-edge deduplication not yet implemented."
            )

        edge_length = _great_circle_lengths(
            node_lon[edge_nodes[:, 0]], node_lat[edge_nodes[:, 0]],
            node_lon[edge_nodes[:, 1]], node_lat[edge_nodes[:, 1]],
        )

        return cls(
            node_lon    = node_lon,
            node_lat    = node_lat,
            edge_nodes  = edge_nodes,
            edge_length = edge_length,
            face_nodes  = face_nodes,
            face_edges  = face_edges,
            face_lon    = face_lon.ravel(),
            face_lat    = face_lat.ravel(),
            area        = area.ravel(),
            mask        = mask.ravel(),
            shape       = shape,
            n_corners   = n_corners,
            attrs       = attrs or {},
        )


def apply_ocean_mask(mesh: GridMesh) -> GridMesh:
    """
    Set mesh.mask from a coastline: 1 over ocean, 0 over land.

    Uses the `global_land_mask` package (pip install global-land-mask).
    The test is on cell centres. Modifies mesh in place and returns it.
    """
    try:
        from global_land_mask import globe
    except ImportError as exc:
        raise ImportError(
            "apply_ocean_mask needs `global_land_mask` "
            "(pip install global-land-mask)"
        ) from exc

    lat = np.rad2deg(mesh.face_lat)
    lon = np.rad2deg(mesh.face_lon)
    lon = (lon + 180.0) % 360.0 - 180.0
    lat = np.clip(lat, -89.999, 89.999)
    mesh.mask = np.where(globe.is_land(lat, lon), 0, 1).astype(np.int32)
    return mesh


def write_to_CF_grid_file(mesh: GridMesh, output_file):
    """
    Write mesh to a CF-1.10 / UGRID-1.0 NetCDF file.

    The rotation angle (angle_of_rotation_from_east_to_x) and its cosine and
    sine are computed from face-centre coordinates and written as face-located
    variables. Currently only supported for 2D mesh.shape.
    """
    rad2deg = 180.0 / np.pi
    angle   = _rotation_angle(mesh.face_lon, mesh.face_lat, mesh.shape)

    ds = xr.Dataset()

    ds["mesh"] = xr.DataArray(
        np.int32(0),
        attrs={
            "cf_role":                "mesh_topology",
            "topology_dimension":     np.int32(2),
            "node_coordinates":       "node_lon node_lat",
            "face_node_connectivity": "face_nodes",
            "face_edge_connectivity": "face_edges",
            "edge_node_connectivity": "edge_nodes",
            "face_coordinates":       "face_lon face_lat",
        },
    )

    ds["node_lon"] = xr.DataArray(
        mesh.node_lon * rad2deg, dims=["nnode"],
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    ds["node_lat"] = xr.DataArray(
        mesh.node_lat * rad2deg, dims=["nnode"],
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )

    ds["face_nodes"] = xr.DataArray(
        mesh.face_nodes.astype(np.int32), dims=["nface", "n_corners"],
        attrs={"cf_role": "face_node_connectivity", "start_index": np.int32(0)},
    )
    ds["face_edges"] = xr.DataArray(
        mesh.face_edges.astype(np.int32), dims=["nface", "n_corners"],
        attrs={"cf_role": "face_edge_connectivity", "start_index": np.int32(0)},
    )
    ds["edge_nodes"] = xr.DataArray(
        mesh.edge_nodes.astype(np.int32), dims=["nedge", "two"],
        attrs={"cf_role": "edge_node_connectivity", "start_index": np.int32(0)},
    )

    ds["face_lon"] = xr.DataArray(
        mesh.face_lon * rad2deg, dims=["nface"],
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    ds["face_lat"] = xr.DataArray(
        mesh.face_lat * rad2deg, dims=["nface"],
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )

    _face = {"mesh": "mesh", "location": "face"}
    _edge = {"mesh": "mesh", "location": "edge"}

    ds["cell_area"] = xr.DataArray(
        mesh.area, dims=["nface"],
        attrs={"standard_name": "cell_area", "units": "m2", **_face},
    )
    ds["mask"] = xr.DataArray(
        mesh.mask.astype(np.int32), dims=["nface"],
        attrs={
            "long_name":     "ocean mask",
            "units":         "1",
            "flag_values":   np.array([0, 1], dtype=np.int32),
            "flag_meanings": "land ocean",
            **_face,
        },
    )
    ds["angle"] = xr.DataArray(
        angle, dims=["nface"],
        attrs={
            "standard_name": "angle_of_rotation_from_east_to_x",
            "units": "radian",
            **_face,
        },
    )
    ds["cos_angle"] = xr.DataArray(
        np.cos(angle), dims=["nface"],
        attrs={"long_name": "cosine of grid rotation angle", "units": "1", **_face},
    )
    ds["sin_angle"] = xr.DataArray(
        np.sin(angle), dims=["nface"],
        attrs={"long_name": "sine of grid rotation angle", "units": "1", **_face},
    )
    ds["edge_length"] = xr.DataArray(
        mesh.edge_length, dims=["nedge"],
        attrs={"long_name": "edge length", "units": "m", **_edge},
    )

    ds.attrs = {
        "Conventions": "CF-1.10, UGRID-1.0",
        "history":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **mesh.attrs,
    }

    ds.to_netcdf(output_file)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_topology_2d(corner_lon, corner_lat, shape, n_corners):
    """
    Exploit structured 2D connectivity to build UGRID arrays without
    floating-point node comparison.

    Corner ordering assumed: [SW, SE, NE, NW] (CCW from lower-left).
    Edge k of face f connects face_nodes[f, k] to face_nodes[f, (k+1) % N]:
      edge 0  SW→SE  bottom, horizontal
      edge 1  SE→NE  right,  vertical
      edge 2  NE→NW  top,    horizontal (stored in SW→SE direction)
      edge 3  NW→SW  left,   vertical   (stored in SW→NW direction)

    Note: i-periodic boundaries are not deduplicated. For a global grid the
    nodes at i=0 and i=ni are distinct in the output even though they represent
    the same geographic points.
    """
    assert n_corners == 4, "_build_topology_2d requires quadrilateral cells"
    nj, ni = shape
    nface  = nj * ni

    f = np.arange(nface)
    j = f // ni
    i = f % ni

    # Unique node index: node(j, i) → j*(ni+1) + i
    face_nodes = np.stack([
        j       * (ni+1) + i,        # SW  k=0
        j       * (ni+1) + (i+1),    # SE  k=1
        (j+1)   * (ni+1) + (i+1),    # NE  k=2
        (j+1)   * (ni+1) + i,        # NW  k=3
    ], axis=1)

    nnode   = (nj+1) * (ni+1)
    node_lon = np.empty(nnode)
    node_lat = np.empty(nnode)
    c_lon   = corner_lon.reshape(nface, n_corners)
    c_lat   = corner_lat.reshape(nface, n_corners)
    for k in range(n_corners):
        node_lon[face_nodes[:, k]] = c_lon[:, k]
        node_lat[face_nodes[:, k]] = c_lat[:, k]

    # Horizontal edges: j ∈ [0, nj], i ∈ [0, ni-1]  →  index = j*ni + i
    n_hedge = (nj+1) * ni
    jh = np.repeat(np.arange(nj+1), ni)
    ih = np.tile(np.arange(ni), nj+1)
    hedge_nodes = np.stack([
        jh * (ni+1) + ih,
        jh * (ni+1) + (ih+1),
    ], axis=1)

    # Vertical edges: j ∈ [0, nj-1], i ∈ [0, ni]  →  index = n_hedge + j*(ni+1) + i
    jv = np.repeat(np.arange(nj), ni+1)
    iv = np.tile(np.arange(ni+1), nj)
    vedge_nodes = np.stack([
        jv       * (ni+1) + iv,
        (jv+1)   * (ni+1) + iv,
    ], axis=1)

    edge_nodes = np.concatenate([hedge_nodes, vedge_nodes], axis=0)

    face_edges = np.stack([
        j       * ni + i,               # bottom  SW→SE
        n_hedge + j * (ni+1) + (i+1),   # right   SE→NE
        (j+1)   * ni + i,               # top     NE→NW  (stored SW→SE)
        n_hedge + j * (ni+1) + i,       # left    NW→SW  (stored SW→NW)
    ], axis=1)

    return node_lon, node_lat, face_nodes, edge_nodes, face_edges


def _great_circle_lengths(lon1, lat1, lon2, lat2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2.0 * _R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _rotation_angle(face_lon, face_lat, shape):
    """
    Angle of the grid i-direction measured anticlockwise from true east.
    Matches CF standard_name angle_of_rotation_from_east_to_x.
    Uses central differences in i with one-sided differences at i-boundaries.
    Only implemented for 2D shape.
    """
    if len(shape) != 2:
        raise NotImplementedError(
            f"Rotation angle only implemented for 2D shape; got {len(shape)}D"
        )
    lon = face_lon.reshape(shape)
    lat = face_lat.reshape(shape)

    dlon = np.empty_like(lon)
    dlat = np.empty_like(lat)

    dlon[:, 1:-1] = lon[:, 2:] - lon[:, :-2]
    dlat[:, 1:-1] = lat[:, 2:] - lat[:, :-2]
    dlon[:, 0]    = lon[:, 1]  - lon[:, 0]
    dlat[:, 0]    = lat[:, 1]  - lat[:, 0]
    dlon[:, -1]   = lon[:, -1] - lon[:, -2]
    dlat[:, -1]   = lat[:, -1] - lat[:, -2]

    dlon = (dlon + np.pi) % (2.0*np.pi) - np.pi
    return np.arctan2(dlat, dlon * np.cos(lat)).ravel()
