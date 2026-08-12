import numpy as np

from EarthSystemGrids.base.UnstructuredGridMesh import UnstructuredGridMesh, _great_circle_lengths


class StructuredQuadMesh(UnstructuredGridMesh):
    """
    Special case of UnstructuredGridMesh: every face is a quadrilateral, and
    the faces are arranged in a logical 2D (nj, ni) grid.

    "Uniform N-sided cells" belongs here, as a specialization -- it exploits
    the structured index arithmetic in `_build_topology_2d` to build
    node/edge connectivity without floating-point coordinate matching, and
    it is what unlocks the rotation angle, which needs a logical
    i-direction to take a centred difference along. General polygon meshes
    use `UnstructuredGridMesh.from_polygons` instead, and carry no `.shape`
    beyond `(nface,)`, so neither shortcut applies to them.
    """

    n_corners = 4

    @classmethod
    def from_corners(cls, corner_lon, corner_lat, face_lon, face_lat,
                     area, mask, shape, attrs=None):
        """
        Build a StructuredQuadMesh from per-face corner coordinate arrays.

        Parameters
        ----------
        corner_lon, corner_lat : (*shape, 4)  radians, CCW from SW corner
        face_lon,   face_lat   : (*shape,)    radians, cell centres
        area                   : (*shape,)    m², cell area
        mask                   : (*shape,)    int, 0=land 1=ocean
        shape                  : logical arrangement, e.g. (nj, ni)
        attrs                  : dict of grid-specific metadata
        """
        if len(shape) != 2:
            raise NotImplementedError(
                f"StructuredQuadMesh.from_corners only supports 2D shape; "
                f"got {len(shape)}D. CubeSphere (3D) requires tile-edge "
                "deduplication not yet implemented."
            )
        node_lon, node_lat, face_nodes, edge_nodes, face_edges = \
            _build_topology_2d(corner_lon, corner_lat, shape)

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
            attrs       = attrs or {},
        )

    def rotation_angle(self):
        """
        Angle of the grid i-direction measured anticlockwise from true east,
        at every face centre. Matches CF standard_name
        angle_of_rotation_from_east_to_x.
        """
        return _rotation_angle(self.face_lon, self.face_lat, self.shape)


def _build_topology_2d(corner_lon, corner_lat, shape):
    """
    Exploit structured 2D quad connectivity to build UGRID arrays without
    floating-point node comparison. Used only by StructuredQuadMesh, where
    every face is known in advance to have exactly 4 corners.

    Corner ordering assumed: [SW, SE, NE, NW] (CCW from lower-left).
    Edge k of face f connects face_nodes[f, k] to face_nodes[f, (k+1) % 4]:
      edge 0  SW→SE  bottom, horizontal
      edge 1  SE→NE  right,  vertical
      edge 2  NE→NW  top,    horizontal (stored in SW→SE direction)
      edge 3  NW→SW  left,   vertical   (stored in SW→NW direction)

    Note: i-periodic boundaries are not deduplicated. For a global grid the
    nodes at i=0 and i=ni are distinct in the output even though they represent
    the same geographic points.
    """
    n_corners = 4
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


def _rotation_angle(face_lon, face_lat, shape):
    """
    Angle of the grid i-direction measured anticlockwise from true east.
    Matches CF standard_name angle_of_rotation_from_east_to_x.
    Uses central differences in i with one-sided differences at i-boundaries.
    Only implemented for 2D shape (i.e. StructuredQuadMesh).
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
