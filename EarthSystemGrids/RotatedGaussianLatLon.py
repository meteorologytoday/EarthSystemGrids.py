import numpy as np
import xarray as xr

from EarthSystemGrids.base import StructuredQuadMesh, _R_EARTH, rotate_lonlat
from EarthSystemGrids import GaussianLatLon


class RotatedGaussianLatLon(StructuredQuadMesh):
    """
    A Gaussian lat-lon grid whose pole is displaced off Earth's true axis
    by an arbitrary rigid rotation.

    Built natively (unrotated -- native pole at the true geographic pole)
    via GaussianLatLon.generate_mesh, then every corner and centre
    coordinate is rigidly rotated about an axis lying in the equatorial
    plane at rotation_axis_longitude_deg, by rotation_deg (right-hand
    rule; see rotate_lonlat).

    In addition to everything StructuredQuadMesh stores, this class keeps
    the grid's native (pre-rotation) 1D cell-centre axes --
    native_lat_deg (nlat,) and native_lon_deg (nlon,), degrees -- as
    `.native_lat_deg` / `.native_lon_deg`, plus the cell-FACE (boundary)
    axes they were derived from -- native_lat_bounds_deg (nlat+1,) and
    native_lon_bounds_deg (nlon+1,) -- as `.native_lat_bounds_deg` /
    `.native_lon_bounds_deg`. A plain Gaussian lat-lon grid is a
    graticule: every row shares one set of longitudes and every column
    shares one set of latitudes, so these are read off directly, before
    rotation, rather than recomputed. Unlike the rotation angle
    (StructuredQuadMesh's extra variable, re-derivable from the rotated
    mesh's own geometry), the native axes are NOT recoverable from the
    rotated face_lon/face_lat alone -- rotation destroys that information
    -- so they are carried as genuine extra state, persisted on write and
    restored on read, rather than recomputed from scratch on every load.
    """

    # Class-level defaults so every method below can read
    # self.native_lat_deg / self.native_lon_deg / self.native_lat_bounds_deg
    # / self.native_lon_bounds_deg unconditionally, without a
    # getattr(..., None) guard: a bare/partially-constructed instance
    # (e.g. mid-from_corners, before generate_mesh/from_SCRIP_file/
    # from_CF_file has had a chance to attach the real arrays) simply
    # sees None here.
    native_lat_deg = None
    native_lon_deg = None
    native_lat_bounds_deg = None
    native_lon_bounds_deg = None

    @classmethod
    def generate_mesh(cls, lat, lon, rotation_axis_longitude_deg, rotation_deg,
                      mask=None, earth_radius: float = _R_EARTH,
                      attrs=None) -> "RotatedGaussianLatLon":
        """
        Build a Gaussian lat-lon grid whose pole is displaced off Earth's
        true axis by an arbitrary rotation.

        Builds the grid natively (unrotated -- native pole at the true
        geographic pole) via GaussianLatLon.generate_mesh, then rigidly
        rotates every corner and centre coordinate about an axis lying in
        the equatorial plane at rotation_axis_longitude_deg, by
        rotation_deg (right-hand rule; see rotate_lonlat).

        Solid angle -- and therefore m² area -- is invariant under a rigid
        rotation of the sphere, so each cell's area from the unrotated mesh
        is reused unchanged rather than recomputed from the rotated
        corners.

        A rotation angle (true east to the grid's local i-direction) is
        not computed here: StructuredQuadMesh.rotation_angle() already
        derives it, exactly, from the rotated mesh's own corner geometry,
        and write_to_SCRIP_grid_file already includes it as grid_angle /
        grid_cos_angle / grid_sin_angle automatically.

        The native (pre-rotation) 1D centre axes are captured from the
        unrotated intermediate arrays, before rotate_lonlat overwrites
        them, and attached to the returned mesh as .native_lat_deg /
        .native_lon_deg, alongside the input face arrays themselves as
        .native_lat_bounds_deg / .native_lon_bounds_deg (also exposed as
        the native_lat / native_lon / native_lat_bounds / native_lon_bounds
        extra variables -- see _compute_extra_variables).

        mask defaults to all-ocean, same convention as
        GaussianLatLon.generate_mesh: land/sea is not rotation-invariant
        (it depends on each cell's absolute, post-rotation location), so
        for a real coastline mask call EarthSystemGrids.base.apply_ocean_mask
        on the returned (already-rotated) mesh -- not before rotating.

        Parameters
        ----------
        lat, lon : as GaussianLatLon.generate_mesh -- explicit cell-FACE
            (boundary) arrays, in the grid's own native (unrotated) frame.
            See GaussianLatLon.gaussian_latitude_bounds / equally_spaced_bounds
            for the standard Gaussian-grid spacing.
        rotation_axis_longitude_deg : degrees. Longitude, in the equatorial
            plane, of the axis the whole grid is rotated about.
        rotation_deg : degrees. Rotation angle about that axis (right-hand
            rule) -- how far the native pole is displaced from the true pole.
        mask         : (nlat, nlon) int, optional, in the *rotated* frame.
            Defaults to all ocean.
        earth_radius : metres, spherical Earth radius used to convert solid
            angle into m² area.
        attrs        : dict of grid-specific metadata.

        Returns
        -------
        RotatedGaussianLatLon, shape (nlat, nlon)
        """
        base_mesh = GaussianLatLon.generate_mesh(lat, lon, earth_radius=earth_radius)
        nlat, nlon = base_mesh.shape

        corner_lon = base_mesh.node_lon[base_mesh.face_nodes].reshape(nlat, nlon, 4)
        corner_lat = base_mesh.node_lat[base_mesh.face_nodes].reshape(nlat, nlon, 4)
        face_lon   = base_mesh.face_lon.reshape(nlat, nlon)
        face_lat   = base_mesh.face_lat.reshape(nlat, nlon)

        # Capture the native (pre-rotation) 1D axes now -- a lat-lon
        # graticule shares one row of longitudes and one column of
        # latitudes, so a single slice each is exact. rotate_lonlat below
        # overwrites the face_lon/face_lat local names with the rotated
        # values, so these must be pulled out first.
        native_lat_deg = np.rad2deg(face_lat[:, 0]).copy()
        native_lon_deg = np.rad2deg(face_lon[0, :]).copy()

        # The face (boundary) arrays are exactly the lat/lon arguments
        # this method was called with -- generate_mesh takes faces, not
        # centres, so no separate derivation is needed here.
        native_lat_bounds_deg = np.asarray(lat, dtype=float).copy()
        native_lon_bounds_deg = np.asarray(lon, dtype=float).copy()

        corner_lon, corner_lat = rotate_lonlat(
            corner_lon, corner_lat, rotation_axis_longitude_deg, rotation_deg)
        face_lon, face_lat = rotate_lonlat(
            face_lon, face_lat, rotation_axis_longitude_deg, rotation_deg)

        if mask is None:
            mask = np.ones((nlat, nlon), dtype=np.int32)

        mesh = cls.from_corners(
            corner_lon = corner_lon,
            corner_lat = corner_lat,
            face_lon   = face_lon,
            face_lat   = face_lat,
            area       = base_mesh.area.reshape(nlat, nlon),
            mask       = np.asarray(mask),
            shape      = (nlat, nlon),
            attrs      = attrs or {},
        )

        # from_corners' cls(...) call only threads through the base
        # UnstructuredGridMesh.__init__ signature, so extra_variables was
        # already computed once (inside __init__) with these two arrays
        # not yet attached. Attach them now and recompute: the hook is a
        # pure function of already-set attributes, the same contract every
        # other construction path in this codebase relies on (see
        # UnstructuredGridMesh._compute_extra_variables), so recomputing
        # it a second time is safe -- just a cheap, vectorized repeat of
        # rotation_angle().
        mesh.native_lat_deg = native_lat_deg
        mesh.native_lon_deg = native_lon_deg
        mesh.native_lat_bounds_deg = native_lat_bounds_deg
        mesh.native_lon_bounds_deg = native_lon_bounds_deg
        mesh.extra_variables = mesh._compute_extra_variables()

        return mesh

    def _native_axis_variables(self):
        """
        {name: xr.DataArray} for native_lat/native_lon and, if present,
        native_lat_bounds/native_lon_bounds, or {} if this instance
        doesn't have the centre axes yet (e.g. built by a generic
        from_corners call rather than generate_mesh/from_SCRIP_file/
        from_CF_file). Single source of truth for both
        _compute_extra_variables (feeds the CF writer) and
        write_to_SCRIP_grid_file's override below, so the dims/attrs are
        defined in exactly one place.
        """
        if self.native_lat_deg is None or self.native_lon_deg is None:
            return {}
        variables = {
            "native_lat": xr.DataArray(
                self.native_lat_deg, dims=["native_lat"],
                attrs={
                    "standard_name": "latitude",
                    "units": "degrees_north",
                    "long_name": "cell-centre latitude on the grid's "
                                  "native (pre-rotation) Gaussian lat-lon axis",
                },
            ),
            "native_lon": xr.DataArray(
                self.native_lon_deg, dims=["native_lon"],
                attrs={
                    "standard_name": "longitude",
                    "units": "degrees_east",
                    "long_name": "cell-centre longitude on the grid's "
                                  "native (pre-rotation) Gaussian lat-lon axis",
                },
            ),
        }
        if self.native_lat_bounds_deg is not None:
            variables["native_lat_bounds"] = xr.DataArray(
                self.native_lat_bounds_deg, dims=["native_lat_bounds"],
                attrs={
                    "standard_name": "latitude",
                    "units": "degrees_north",
                    "long_name": "cell-face latitude on the grid's "
                                  "native (pre-rotation) Gaussian lat-lon axis",
                },
            )
        if self.native_lon_bounds_deg is not None:
            variables["native_lon_bounds"] = xr.DataArray(
                self.native_lon_bounds_deg, dims=["native_lon_bounds"],
                attrs={
                    "standard_name": "longitude",
                    "units": "degrees_east",
                    "long_name": "cell-face longitude on the grid's "
                                  "native (pre-rotation) Gaussian lat-lon axis",
                },
            )
        return variables

    def _compute_extra_variables(self):
        """
        StructuredQuadMesh's angle/cos_angle/sin_angle, plus native_lat/
        native_lon on top. Unlike the angle fields, the native axes are
        only present once generate_mesh/from_SCRIP_file/from_CF_file has
        attached native_lat_deg/native_lon_deg (see the class-level
        defaults above); a bare cls(...) construction (e.g. mid-
        from_corners) contributes only the inherited angle fields.
        """
        extra = dict(super()._compute_extra_variables())
        extra.update(self._native_axis_variables())
        return extra

    def write_to_SCRIP_grid_file(self, output_file, flatten: bool = True):
        """
        Writes exactly what StructuredQuadMesh.write_to_SCRIP_grid_file
        writes (via super()), then adds native_lat/native_lon (cell
        centres, nlat/nlon) and native_lat_bounds/native_lon_bounds (cell
        faces, nlat+1/nlon+1) on top, each under its own dedicated
        dimension -- independent of `flatten`, so the extension schema
        doesn't change shape depending on that flag, and from_SCRIP_file
        below has one reader path for both. ESMF_RegridWeightGen/ncremap
        ignore variables they don't recognise, so this stays a valid
        SCRIP file.

        The parent writer builds and writes its xr.Dataset in one step
        and returns None, so it can't be intercepted before write; the
        extra variables are added by reading the just-written file back
        in fully, merging, and rewriting once.
        """
        super().write_to_SCRIP_grid_file(output_file, flatten=flatten)

        extra = self._native_axis_variables()
        if not extra:
            return

        with xr.open_dataset(output_file, mask_and_scale=False) as ds:
            ds = ds.load()
        ds.update(extra)
        ds.to_netcdf(output_file)

    @staticmethod
    def _restore_native_axes(mesh, ds):
        """
        Read native_lat/native_lon and, if present, native_lat_bounds/
        native_lon_bounds back from an already-open dataset onto mesh,
        then recompute mesh.extra_variables so they show up there too.
        Shared by from_SCRIP_file and from_CF_file below.
        """
        if "native_lat" in ds.variables and "native_lon" in ds.variables:
            mesh.native_lat_deg = ds["native_lat"].values.copy()
            mesh.native_lon_deg = ds["native_lon"].values.copy()
        if "native_lat_bounds" in ds.variables and "native_lon_bounds" in ds.variables:
            mesh.native_lat_bounds_deg = ds["native_lat_bounds"].values.copy()
            mesh.native_lon_bounds_deg = ds["native_lon_bounds"].values.copy()
        mesh.extra_variables = mesh._compute_extra_variables()
        return mesh

    @classmethod
    def from_SCRIP_file(cls, input_file):
        """
        super().from_SCRIP_file (StructuredQuadMesh's) builds the mesh via
        from_corners, same as any other structured quad grid; this then
        separately restores the native axes if present (they won't be,
        for a SCRIP file this class didn't write itself) via
        _restore_native_axes.
        """
        mesh = super().from_SCRIP_file(input_file)
        with xr.open_dataset(input_file, mask_and_scale=False) as ds:
            cls._restore_native_axes(mesh, ds)
        return mesh

    @classmethod
    def from_CF_file(cls, input_file):
        """
        Mirrors from_SCRIP_file above, for the CF/UGRID writer. This is
        the one place in this class that deliberately departs from the
        base classes' "never trust extra_variables from the file, always
        recompute from geometry" rule (see
        UnstructuredGridMesh._compute_extra_variables): the native axes
        cannot be re-derived from the rotated face_lon/lat alone --
        rotation is lossy -- so they are read back from the file rather
        than recomputed.
        """
        mesh = super().from_CF_file(input_file)
        with xr.open_dataset(input_file, mask_and_scale=False) as ds:
            cls._restore_native_axes(mesh, ds)
        return mesh
