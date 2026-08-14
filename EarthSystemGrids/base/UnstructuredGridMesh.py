from datetime import datetime, timezone

import numpy as np
import xarray as xr


_R_EARTH     = 6.371e6  # m, spherical Earth
_FILL_VALUE  = -1       # sentinel for unused face_nodes / face_edges slots
_FACE_ATTRS  = {"mesh": "mesh", "location": "face"}
_EDGE_ATTRS  = {"mesh": "mesh", "location": "edge"}


class UnstructuredGridMesh:
    """
    UGRID-native storage for a mesh of arbitrary polygonal faces.

    Faces need not share a corner count -- triangles, quadrilaterals,
    pentagons and so on can coexist in the same mesh. face_nodes and
    face_edges are stored as (nface, max_n_corners) arrays, padded with
    _FILL_VALUE wherever a face has fewer corners than the widest face in
    the mesh. This is exactly how the UGRID convention expresses a ragged
    face_node_connectivity in a dense NetCDF array (a `_FillValue`
    attribute on the variable).

    A structured, uniform-N-sided grid (a lat-lon or curvilinear quad mesh)
    is a special case of this storage, not a different one: see
    `StructuredQuadMesh`, which adds the (nj, ni) index-based topology
    builder and the structured-only rotation angle.

    All angular quantities are in radians. All linear quantities are in
    metres. The 1D storage (nnode / nedge / nface) is the canonical form;
    `.shape` records the logical arrangement, if the mesh has one.

    Naming: these are UGRID connectivity arrays, and each name is short for
    "X_Y_connectivity" -- read `face_nodes` as "for each face, its nodes"
    and `face_edges` as "for each face, its edges", the same way UGRID
    itself names `face_node_connectivity` / `face_edge_connectivity` /
    `edge_node_connectivity`. Concretely:

      face_nodes : face -> node.  Row f lists the node indices (corners)
                   that bound face f, in CCW order.
      edge_nodes : edge -> node.  Row e lists the 2 node indices that are
                   edge e's endpoints.
      face_edges : face -> edge.  Row f lists the edge indices that bound
                   face f, in the same order as face_nodes[f]: entry k is
                   the edge from face_nodes[f, k] to face_nodes[f, (k+1) % n_f].

    Attributes
    ----------
    node_lon, node_lat : (nnode,)                unique corner coordinates, radians
    edge_nodes         : (nedge, 2) int           edge -> node: the 2 endpoint
                                                   node indices of each edge
    edge_length        : (nedge,)                 great-circle length, metres
    face_nodes         : (nface, max_n_corners)   face -> node: the corner node
                                                   indices of each face, CCW,
                                                   padded with _FILL_VALUE past
                                                   that face's own corner count
    face_edges         : (nface, max_n_corners)   face -> edge: the bounding edge
                                                   indices of each face, padded
                                                   the same way; edge k connects
                                                   face_nodes[f, k] to
                                                   face_nodes[f, (k+1) % n_f]
                                                   where n_f is face f's corner count
    face_lon, face_lat : (nface,)                 cell-centre coordinates, radians
    area               : (nface,)                 cell area, m²
    mask               : (nface,) int             0 = land, 1 = ocean
    shape              : tuple                    logical arrangement, e.g.
                                                   (nj, ni) if structured,
                                                   else (nface,)
    attrs              : dict                     grid-specific metadata
    extra_variables    : dict                     {name: xr.DataArray} of any
                                                   subclass-specific CF variables
                                                   (e.g. StructuredQuadMesh's
                                                   rotation angle), computed once
                                                   by _compute_extra_variables()
                                                   at construction time -- see
                                                   that method
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
        self.attrs       = attrs or {}
        # computed last, since it may depend on every attribute above (e.g.
        # StructuredQuadMesh's override calls rotation_angle(), which needs
        # face_nodes/node_lon/node_lat/face_lon/face_lat already set)
        self.extra_variables = self._compute_extra_variables()

    @property
    def max_n_corners(self):
        """Width of the padded face_nodes / face_edges arrays."""
        return self.face_nodes.shape[1]

    @property
    def n_nodes_per_face(self):
        """(nface,) int: each face's true corner count, ignoring padding."""
        return np.count_nonzero(self.face_nodes != _FILL_VALUE, axis=1)

    def _compute_extra_variables(self):
        """
        Hook for subclasses to contribute subclass-specific CF variables.

        Called once, at the end of __init__, and the result stored as
        `self.extra_variables` -- a plain dict, not a method to call again
        later. This is what makes those fields (e.g. StructuredQuadMesh's
        rotation angle) come back automatically on every construction path,
        including from_CF_file: loading a mesh just calls cls(...) like any
        other constructor, so this hook reruns and recomputes them fresh
        from the loaded geometry, rather than trusting whatever a NetCDF
        file happened to have stored under those names.

        A subclass overriding this may add fields (e.g. an angle that needs
        a logical i-direction) or replace a base variable outright by
        returning its name again; write_to_CF_grid_file applies
        self.extra_variables last, after every base CF/UGRID variable.
        UnstructuredGridMesh itself has no subclass-specific fields, so this
        is empty.

        Returns
        -------
        dict of {variable_name: xr.DataArray}
        """
        return {}

    def write_to_CF_grid_file(self, output_file):
        """
        Write this mesh to a CF-1.10 / UGRID-1.0 NetCDF file.

        Where a face's corner count is below max_n_corners, the padding
        slots in face_nodes / face_edges are marked with a `_FillValue`,
        per the UGRID convention for meshes that do not have a uniform
        face arity.

        Fields that only some mesh subclasses can define -- e.g. a
        rotation angle, which needs a logical i-direction -- are not
        handled here at all. They come from `self.extra_variables`, a dict
        already computed at construction time; see
        UnstructuredGridMesh._compute_extra_variables and
        StructuredQuadMesh._compute_extra_variables.
        """
        rad2deg = 180.0 / np.pi

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
            self.node_lon * rad2deg, dims=["nnode"],
            attrs={"standard_name": "longitude", "units": "degrees_east"},
        )
        ds["node_lat"] = xr.DataArray(
            self.node_lat * rad2deg, dims=["nnode"],
            attrs={"standard_name": "latitude", "units": "degrees_north"},
        )

        ragged = bool(np.any(self.face_nodes == _FILL_VALUE))

        ds["face_nodes"] = xr.DataArray(
            self.face_nodes.astype(np.int32), dims=["nface", "max_face_nodes"],
            attrs={"cf_role": "face_node_connectivity", "start_index": np.int32(0)},
        )
        ds["face_edges"] = xr.DataArray(
            self.face_edges.astype(np.int32), dims=["nface", "max_face_nodes"],
            attrs={"cf_role": "face_edge_connectivity", "start_index": np.int32(0)},
        )
        # _FillValue must travel via encoding, not attrs, or xarray raises on write.
        ds["face_nodes"].encoding["_FillValue"] = np.int32(_FILL_VALUE) if ragged else None
        ds["face_edges"].encoding["_FillValue"] = np.int32(_FILL_VALUE) if ragged else None

        ds["edge_nodes"] = xr.DataArray(
            self.edge_nodes.astype(np.int32), dims=["nedge", "two"],
            attrs={"cf_role": "edge_node_connectivity", "start_index": np.int32(0)},
        )

        ds["face_lon"] = xr.DataArray(
            self.face_lon * rad2deg, dims=["nface"],
            attrs={"standard_name": "longitude", "units": "degrees_east"},
        )
        ds["face_lat"] = xr.DataArray(
            self.face_lat * rad2deg, dims=["nface"],
            attrs={"standard_name": "latitude", "units": "degrees_north"},
        )

        ds["cell_area"] = xr.DataArray(
            self.area, dims=["nface"],
            attrs={"standard_name": "cell_area", "units": "m2", **_FACE_ATTRS},
        )
        ds["mask"] = xr.DataArray(
            self.mask.astype(np.int32), dims=["nface"],
            attrs={
                "long_name":     "ocean mask",
                "units":         "1",
                "flag_values":   np.array([0, 1], dtype=np.int32),
                "flag_meanings": "land ocean",
                **_FACE_ATTRS,
            },
        )

        ds["edge_length"] = xr.DataArray(
            self.edge_length, dims=["nedge"],
            attrs={"long_name": "edge length", "units": "m", **_EDGE_ATTRS},
        )

        ds.update(self.extra_variables)

        ds.attrs = {
            "Conventions": "CF-1.10, UGRID-1.0",
            "history":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "shape":       np.array(self.shape, dtype=np.int64),
            **self.attrs,
        }

        ds.to_netcdf(output_file)

    @classmethod
    def from_CF_file(cls, input_file):
        """
        Load a mesh back from a file written by write_to_CF_grid_file.

        This is a round-trip counterpart to that writer, not a general
        UGRID-1.0 reader: it expects exactly the variable names, units and
        the custom `shape` attribute write_to_CF_grid_file produces, rather
        than following node_coordinates / face_node_connectivity etc.
        indirection to support arbitrary UGRID files from other tools.

        Called on StructuredQuadMesh, this constructs a StructuredQuadMesh
        (see its override, which adds a validity check); called on
        UnstructuredGridMesh directly, it constructs the base class. Either
        way, extra_variables (e.g. StructuredQuadMesh's rotation angle) are
        never read from the file even if present -- construction always
        calls cls(...), which recomputes them fresh from the loaded
        geometry via _compute_extra_variables(), the same as any other
        constructor.
        """
        ds = xr.open_dataset(input_file, mask_and_scale=False)

        deg2rad = np.pi / 180.0
        # a length-1 array attribute (shape == (nface,)) round-trips through
        # NetCDF as a bare scalar, not a 1-element array -- atleast_1d handles both
        shape = tuple(int(n) for n in np.atleast_1d(ds.attrs["shape"]))
        attrs = {k: v for k, v in ds.attrs.items()
                 if k not in ("Conventions", "history", "shape")}

        return cls(
            node_lon    = ds["node_lon"].values * deg2rad,
            node_lat    = ds["node_lat"].values * deg2rad,
            edge_nodes  = ds["edge_nodes"].values,
            edge_length = ds["edge_length"].values,
            face_nodes  = ds["face_nodes"].values,
            face_edges  = ds["face_edges"].values,
            face_lon    = ds["face_lon"].values * deg2rad,
            face_lat    = ds["face_lat"].values * deg2rad,
            area        = ds["cell_area"].values,
            mask        = ds["mask"].values,
            shape       = shape,
            attrs       = attrs,
        )

    @classmethod
    def from_polygons(cls, face_corner_lon, face_corner_lat, face_lon, face_lat,
                      area, mask, attrs=None, node_atol=1e-10):
        """
        Build an UnstructuredGridMesh from a ragged, per-face list of corner
        coordinates.

        This is the true UGRID-native path: faces are not required to share
        a corner count, so a mixed mesh -- e.g. triangles closing off a
        polar cap of an otherwise quadrilateral grid -- is a first-class
        input, not a special case. Shared nodes and edges between
        neighbouring faces are found by coordinate matching within
        `node_atol`, since there is no structured index arithmetic to fall
        back on.

        Parameters
        ----------
        face_corner_lon, face_corner_lat : sequence of length nface, each a
            1-D array of that face's corner coordinates in radians, CCW.
            Different faces may have different lengths.
        face_lon, face_lat : (nface,) radians, cell centres
        area                : (nface,) m², cell area
        mask                 : (nface,) int, 0=land 1=ocean
        attrs                : dict of grid-specific metadata
        node_atol            : float, radians. Corners closer than this in
                               both lon and lat are treated as the same node.

        Notes
        -----
        O(nface * max_n_corners) in Python, since node matching cannot use
        the structured shortcut in StructuredQuadMesh. Fine for the
        thousands-to-low-millions of faces typical of a global mesh; expect
        it to dominate runtime well before that.
        """
        nface = len(face_corner_lon)
        if len(face_corner_lat) != nface:
            raise ValueError(
                "face_corner_lon and face_corner_lat must have the same length")

        node_lon_list = []
        node_lat_list = []
        node_index    = {}

        def get_node(lon, lat):
            key = (round(lon / node_atol), round(lat / node_atol))
            idx = node_index.get(key)
            if idx is None:
                idx = len(node_lon_list)
                node_index[key] = idx
                node_lon_list.append(lon)
                node_lat_list.append(lat)
            return idx

        n_per_face = np.array([len(c) for c in face_corner_lon], dtype=np.int64)
        if np.any(n_per_face < 3):
            raise ValueError("every face needs at least 3 corners")
        max_n = int(n_per_face.max())

        face_nodes = np.full((nface, max_n), _FILL_VALUE, dtype=np.int64)
        for f in range(nface):
            lons, lats = face_corner_lon[f], face_corner_lat[f]
            for k in range(n_per_face[f]):
                face_nodes[f, k] = get_node(float(lons[k]), float(lats[k]))

        node_lon = np.array(node_lon_list)
        node_lat = np.array(node_lat_list)

        edge_index      = {}
        edge_nodes_list = []
        face_edges      = np.full((nface, max_n), _FILL_VALUE, dtype=np.int64)
        for f in range(nface):
            n = n_per_face[f]
            for k in range(n):
                a = int(face_nodes[f, k])
                b = int(face_nodes[f, (k+1) % n])
                key = (a, b) if a < b else (b, a)
                eidx = edge_index.get(key)
                if eidx is None:
                    eidx = len(edge_nodes_list)
                    edge_index[key] = eidx
                    edge_nodes_list.append((a, b))
                face_edges[f, k] = eidx

        edge_nodes = np.array(edge_nodes_list, dtype=np.int64)

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
            face_lon    = np.asarray(face_lon).ravel(),
            face_lat    = np.asarray(face_lat).ravel(),
            area        = np.asarray(area).ravel(),
            mask        = np.asarray(mask).ravel(),
            shape       = (nface,),
            attrs       = attrs or {},
        )

    @classmethod
    def from_SCRIP_file(cls, input_file, node_atol=1e-10):
        """
        Load a mesh from a SCRIP grid file.

        This is the from_polygons analogue of from_SCRIP_file, above: a
        general reader, not tied to any writer in this package (there is
        no UnstructuredGridMesh.write_to_SCRIP_grid_file -- only
        StructuredQuadMesh has one). It makes no assumption that every
        face has the same true corner count or that the file has a 2D
        logical shape, unlike StructuredQuadMesh.from_SCRIP_file.

        SCRIP's grid_corner_lon/lat arrays are always rectangular
        (grid_size, grid_corners), so a face with fewer real corners than
        grid_corners is recovered by trimming trailing corners that
        duplicate the previous one -- the common SCRIP convention for
        mixed-arity meshes (e.g. MPAS output) -- before handing the ragged
        per-face corner lists to from_polygons for floating-point
        node/edge matching. grid_dims / grid_rank are not used to recover
        a logical shape; the resulting mesh always has shape == (nface,),
        same as from_polygons.

        Called on StructuredQuadMesh, this method is NOT reused: that
        class overrides from_SCRIP_file with its own implementation
        rather than delegating to this one via super(), because the mesh
        this method builds -- 1D shape, corners matched by float
        proximity -- doesn't satisfy what StructuredQuadMesh needs (a 2D
        (nj, ni) shape and the structured SW/SE/NE/NW connectivity that
        rotation_angle relies on). StructuredQuadMesh.from_SCRIP_file
        takes the fast structured-index path via from_corners instead of
        the O(nface) floating-point matching here.

        Parameters
        ----------
        input_file : path to a SCRIP grid file
        node_atol  : float, radians. Corners within this tolerance of the
                     previous one are treated as trailing padding rather
                     than a real corner, and also passed through to
                     from_polygons for shared-node matching.
        """
        ds = xr.open_dataset(input_file, mask_and_scale=False)

        def _to_radians(da):
            units = da.attrs.get("units", "radians")
            values = da.values
            return values * (np.pi / 180.0) if units.lower().startswith("deg") else values

        face_lon   = _to_radians(ds["grid_center_lon"])
        face_lat   = _to_radians(ds["grid_center_lat"])
        corner_lon = _to_radians(ds["grid_corner_lon"])
        corner_lat = _to_radians(ds["grid_corner_lat"])
        area       = ds["grid_area"].values * _R_EARTH**2
        mask       = ds["grid_imask"].values

        face_corner_lon = []
        face_corner_lat = []
        for lons, lats in zip(corner_lon, corner_lat):
            n = len(lons)
            while n > 3 and abs(lons[n-1] - lons[n-2]) < node_atol and abs(lats[n-1] - lats[n-2]) < node_atol:
                n -= 1
            face_corner_lon.append(lons[:n])
            face_corner_lat.append(lats[:n])

        attrs = {}
        if ds.attrs.get("title"):
            attrs["title"] = ds.attrs["title"]

        return cls.from_polygons(
            face_corner_lon = face_corner_lon,
            face_corner_lat = face_corner_lat,
            face_lon        = face_lon,
            face_lat        = face_lat,
            area            = area,
            mask            = mask,
            attrs           = attrs,
            node_atol       = node_atol,
        )


def apply_ocean_mask(mesh: UnstructuredGridMesh) -> UnstructuredGridMesh:
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


def _great_circle_lengths(lon1, lat1, lon2, lat2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2.0 * _R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
