import numpy as np
import xarray as xr

from EarthSystemGrids.base.UnstructuredGridMesh import (
    UnstructuredGridMesh, _great_circle_lengths, _FACE_ATTRS, _R_EARTH,
)

_VECTOR_ROTATION_COMMENT = (
    "'angle' is the angle (radians, positive anticlockwise) from true east "
    "to the grid's local i-direction at each cell centre; 'cos_angle' / "
    "'sin_angle' are its cosine/sine, provided so consumers don't have to "
    "re-derive them (named grid_angle/grid_cos_angle/grid_sin_angle in the "
    "SCRIP writer, angle/cos_angle/sin_angle in the CF writer). To rotate "
    "a vector with grid-relative components (u_grid along i, v_grid along "
    "j) into geographic east/north components: "
    "u_east = u_grid*cos_angle - v_grid*sin_angle; "
    "v_north = u_grid*sin_angle + v_grid*cos_angle. "
    "To go the other way, east/north into grid-relative: "
    "u_grid = u_east*cos_angle + v_north*sin_angle; "
    "v_grid = -u_east*sin_angle + v_north*cos_angle."
)

_ANGLE_ATTRS = {
    "long_name": "angle of rotation from true east to grid i-direction",
    "comment": _VECTOR_ROTATION_COMMENT,
}
_COS_ANGLE_ATTRS = {"long_name": "cosine of grid rotation angle"}
_SIN_ANGLE_ATTRS = {"long_name": "sine of grid rotation angle"}


class StructuredQuadMesh(UnstructuredGridMesh):
    """
    Special case of UnstructuredGridMesh: every face is a quadrilateral, and
    the faces are arranged in a logical 2D (nj, ni) grid.

    "Uniform N-sided cells" belongs here, as a specialization -- it exploits
    the structured index arithmetic in `_build_topology_2d` to build
    node/edge connectivity without floating-point coordinate matching, and
    it is what unlocks the rotation angle, which needs each face to have a
    well-defined bottom/top edge pair (the i-direction) to estimate a
    tangent from. General polygon meshes use `UnstructuredGridMesh.from_polygons`
    instead, and carry no `.shape` beyond `(nface,)`, so neither shortcut
    applies to them.
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

    def write_to_SCRIP_grid_file(self, output_file, flatten: bool = True):
        """
        Write this mesh as a SCRIP grid file, readable by
        ESMF_RegridWeightGen (and by extension ncremap, which wraps it).

        With flatten=True the cells are collapsed onto a single
        `grid_size` dimension and written in degrees, which is the
        portable form. With flatten=False the (lat, lon) structure is
        kept and radians are used, which is easier to inspect but relies
        on the reader honouring grid_dims.

        Also includes `grid_angle` (radians, always -- not degree-converted
        under flatten=True, same convention as grid_area) plus its cosine
        and sine, `grid_cos_angle` / `grid_sin_angle`: the rotation angle
        from true east to the grid's i-direction, from rotation_angle().
        None of the three are part of the classic SCRIP spec, but several
        models that consume SCRIP files (for rotating vector components
        onto the grid) expect angle fields under these names.
        """
        import xarray as xr

        nj, ni = self.shape
        grid_corners   = self.n_corners
        grid_dims      = [ni, nj]   # ESMF reads in reverse array order; undocumented
        grid_dim_names = ["lat", "lon"]
        rad2deg        = 180.0 / np.pi

        corner_lon = self.node_lon[self.face_nodes]  # (nface, N)
        corner_lat = self.node_lat[self.face_nodes]
        area_sr    = self.area / _R_EARTH**2          # m² → steradians for SCRIP
        angle      = self.rotation_angle()
        cos_angle  = np.cos(angle)
        sin_angle  = np.sin(angle)

        if flatten:
            ds = xr.Dataset(
                data_vars=dict(
                    grid_dims       = (["grid_rank"],                  grid_dims),
                    grid_imask      = (["grid_size"],                  self.mask),
                    grid_center_lat = (["grid_size"],                  self.face_lat * rad2deg, {"units": "degrees"}),
                    grid_center_lon = (["grid_size"],                  self.face_lon * rad2deg, {"units": "degrees"}),
                    grid_corner_lat = (["grid_size", "grid_corners"],  corner_lat    * rad2deg, {"units": "degrees"}),
                    grid_corner_lon = (["grid_size", "grid_corners"],  corner_lon    * rad2deg, {"units": "degrees"}),
                    grid_area       = (["grid_size"],                  area_sr,                 {"units": "radians^2"}),
                    grid_angle      = (["grid_size"],                  angle,                   {"units": "radians", **_ANGLE_ATTRS}),
                    grid_cos_angle  = (["grid_size"],                  cos_angle,               {"units": "1", **_COS_ANGLE_ATTRS}),
                    grid_sin_angle  = (["grid_size"],                  sin_angle,               {"units": "1", **_SIN_ANGLE_ATTRS}),
                ),
            )
        else:
            ds = xr.Dataset(
                data_vars=dict(
                    grid_dims       = (["grid_rank"],                              grid_dims),
                    grid_imask      = ([*grid_dim_names],                          self.mask.reshape(self.shape)),
                    grid_center_lat = ([*grid_dim_names],                          self.face_lat.reshape(self.shape), {"units": "radians"}),
                    grid_center_lon = ([*grid_dim_names],                          self.face_lon.reshape(self.shape), {"units": "radians"}),
                    grid_corner_lat = ([*grid_dim_names, "grid_corners"],          corner_lat.reshape(*self.shape, grid_corners),   {"units": "radians"}),
                    grid_corner_lon = ([*grid_dim_names, "grid_corners"],          corner_lon.reshape(*self.shape, grid_corners),   {"units": "radians"}),
                    grid_area       = ([*grid_dim_names],                          area_sr.reshape(self.shape),     {"units": "radians^2"}),
                    grid_angle      = ([*grid_dim_names],                          angle.reshape(self.shape),       {"units": "radians", **_ANGLE_ATTRS}),
                    grid_cos_angle  = ([*grid_dim_names],                          cos_angle.reshape(self.shape),   {"units": "1", **_COS_ANGLE_ATTRS}),
                    grid_sin_angle  = ([*grid_dim_names],                          sin_angle.reshape(self.shape),   {"units": "1", **_SIN_ANGLE_ATTRS}),
                ),
            )

        ds.attrs["title"] = self.attrs.get("title", "")
        ds.attrs["comment"] = _VECTOR_ROTATION_COMMENT
        ds.to_netcdf(output_file)

    @classmethod
    def from_SCRIP_file(cls, input_file):
        """
        Load a StructuredQuadMesh back from a file written by
        write_to_SCRIP_grid_file.

        Round-trip counterpart to that writer, not a general SCRIP reader:
        it assumes grid_dims = [ni, nj] (ESMF's reversed order -- see the
        writer's docstring) and SW/SE/NE/NW corner ordering per face, the
        same assumptions _build_topology_2d makes when from_corners (below)
        hands it these corner arrays. A SCRIP file from another tool, with
        a different corner winding or grid_rank, will load without error
        but build a mesh with scrambled topology.

        Both of the writer's two layouts -- flattened (grid_size, degrees)
        and unflattened ((lat, lon), radians) -- reshape to the same
        (nj, ni, ...) arrays via a plain `.reshape`, since both already lay
        their data out row-major over (j, i); no explicit branch on layout
        is needed. Units are read from each variable's own `units`
        attribute rather than assumed from the layout.

        grid_area is converted back from steradians (grid_area * R_EARTH**2)
        to match `.area`'s m² convention. grid_angle / grid_cos_angle /
        grid_sin_angle are not read even if present -- like every
        `extra_variables` field, the rotation angle is recomputed from
        geometry by StructuredQuadMesh._compute_extra_variables, not
        trusted from the file.

        Deliberately does not call UnstructuredGridMesh.from_SCRIP_file via
        super(): that method matches corners by floating-point proximity
        and returns a 1D (nface,) shape, neither of which satisfies what
        this class needs -- a 2D (nj, ni) shape and the structured
        SW/SE/NE/NW connectivity rotation_angle relies on. This method
        takes the cheaper structured-index path via from_corners instead.
        """
        ds = xr.open_dataset(input_file, mask_and_scale=False)

        ni, nj = (int(n) for n in ds["grid_dims"].values)
        shape  = (nj, ni)

        grid_corners = ds.sizes.get("grid_corners")
        if grid_corners != cls.n_corners:
            raise ValueError(
                f"{input_file!r} has grid_corners={grid_corners}, not "
                f"{cls.n_corners} -- not a uniform quadrilateral SCRIP grid."
            )

        def _to_radians(da):
            units = da.attrs.get("units", "radians")
            values = da.values
            return values * (np.pi / 180.0) if units.lower().startswith("deg") else values

        face_lon   = _to_radians(ds["grid_center_lon"]).reshape(shape)
        face_lat   = _to_radians(ds["grid_center_lat"]).reshape(shape)
        corner_lon = _to_radians(ds["grid_corner_lon"]).reshape(*shape, grid_corners)
        corner_lat = _to_radians(ds["grid_corner_lat"]).reshape(*shape, grid_corners)
        area       = ds["grid_area"].values.reshape(shape) * _R_EARTH**2
        mask       = ds["grid_imask"].values.reshape(shape)

        attrs = {}
        if ds.attrs.get("title"):
            attrs["title"] = ds.attrs["title"]

        return cls.from_corners(
            corner_lon = corner_lon,
            corner_lat = corner_lat,
            face_lon   = face_lon,
            face_lat   = face_lat,
            area       = area,
            mask       = mask,
            shape      = shape,
            attrs      = attrs,
        )

    @classmethod
    def from_CF_file(cls, input_file):
        """
        Load a StructuredQuadMesh back from a file written by
        write_to_CF_grid_file.

        See UnstructuredGridMesh.from_CF_file for what's actually read;
        this just checks the result is a valid StructuredQuadMesh --
        every face has exactly 4 corners and the logical shape is 2D --
        rather than silently handing back an object that would fail later,
        confusingly, inside rotation_angle() or write_to_SCRIP_grid_file.
        """
        mesh = super().from_CF_file(input_file)
        if mesh.max_n_corners != 4 or np.any(mesh.n_nodes_per_face != 4):
            raise ValueError(
                f"{input_file!r} is not a uniform quadrilateral mesh -- "
                "load it with UnstructuredGridMesh.from_CF_file instead."
            )
        if len(mesh.shape) != 2:
            raise ValueError(
                f"{input_file!r} has shape {mesh.shape}, not a 2D (nj, ni) "
                "logical arrangement."
            )
        return mesh

    def rotation_angle(self):
        """
        Angle of the grid i-direction measured anticlockwise from true east,
        at every face centre. Matches CF standard_name
        angle_of_rotation_from_east_to_x.

        Computed from each face's own SW/SE/NE/NW corners, not by
        differencing neighbouring face centres. The corners are converted
        to Cartesian unit vectors on the sphere; averaging the bottom edge
        (SW->SE) and top edge (NW->NE) -- both of which run in the +i
        direction -- estimates the local i-direction tangent, which is
        then resolved against the true local east/north basis at the face
        centre via a dot product.

        This is exact spherical geometry, not the small-angle / flat-Earth
        approximation a lon/lat finite difference scaled by cos(lat) would
        be, so it stays correct near poles and in tightly refined regions.
        It is also well-defined per face regardless of neighbours -- no
        one-sided differencing needed at grid boundaries -- and has no
        antimeridian seam to special-case, since Cartesian coordinates have
        no seam to cross.
        """
        corner_lon = self.node_lon[self.face_nodes]   # (nface, 4): SW, SE, NE, NW
        corner_lat = self.node_lat[self.face_nodes]
        p = _unit_vectors(corner_lon, corner_lat)      # (nface, 4, 3)

        # local i-direction tangent: average of the bottom edge (SW->SE) and
        # top edge (NW->NE), both of which run in the +i direction
        v = 0.5 * ((p[:, 1] - p[:, 0]) + (p[:, 2] - p[:, 3]))

        lon0, lat0 = self.face_lon, self.face_lat
        e_east  = np.stack([-np.sin(lon0), np.cos(lon0), np.zeros_like(lon0)], axis=-1)
        e_north = np.stack([
            -np.sin(lat0) * np.cos(lon0),
            -np.sin(lat0) * np.sin(lon0),
            np.cos(lat0),
        ], axis=-1)

        cos_component = np.sum(v * e_east,  axis=-1)
        sin_component = np.sum(v * e_north, axis=-1)
        return np.arctan2(sin_component, cos_component)

    def _compute_extra_variables(self):
        """
        Contributes the rotation angle and its cosine/sine, computed once
        at construction time (see UnstructuredGridMesh.__init__) and stored
        in self.extra_variables for write_to_CF_grid_file to use. Meaningful
        only here: they need each face to have a well-defined i-direction
        edge pair, which only a structured quad mesh guarantees.
        """
        angle = self.rotation_angle()
        return {
            "angle": xr.DataArray(
                angle, dims=["nface"],
                attrs={
                    "standard_name": "angle_of_rotation_from_east_to_x",
                    "units": "radian",
                    **_ANGLE_ATTRS,
                    **_FACE_ATTRS,
                },
            ),
            "cos_angle": xr.DataArray(
                np.cos(angle), dims=["nface"],
                attrs={"units": "1", **_COS_ANGLE_ATTRS, **_FACE_ATTRS},
            ),
            "sin_angle": xr.DataArray(
                np.sin(angle), dims=["nface"],
                attrs={"units": "1", **_SIN_ANGLE_ATTRS, **_FACE_ATTRS},
            ),
        }


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

    Returns
    -------
    node_lon, node_lat : (nnode,)         nnode = (nj+1)*(ni+1)
    face_nodes         : (nface, 4) int   nface = nj*ni; row-major over (j, i)
    edge_nodes         : (nedge, 2) int   nedge = n_hedge + n_vedge, where
                                           n_hedge = (nj+1)*ni, n_vedge = nj*(ni+1)
    face_edges         : (nface, 4) int   indices into edge_nodes above

    edge_nodes is laid out as two contiguous blocks, NOT interleaved per
    face -- every horizontal edge in the mesh, then every vertical edge:

      [0, n_hedge)             horizontal edges, index = j*ni + i,
                                j in [0, nj] (all nj+1 row-lines),
                                i in [0, ni-1] (gaps along each row-line)
      [n_hedge, n_hedge+n_vedge)  vertical edges, index = n_hedge + j*(ni+1) + i,
                                j in [0, nj-1] (each of the nj rows),
                                i in [0, ni] (all ni+1 column-lines per row)

    face_edges' bottom/top entries index into the horizontal block; its
    right/left entries add the n_hedge offset to reach into the vertical
    block.
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


def _unit_vectors(lon, lat):
    """Cartesian unit vectors on the sphere from radian lon/lat, stacked on the last axis."""
    return np.stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ], axis=-1)
