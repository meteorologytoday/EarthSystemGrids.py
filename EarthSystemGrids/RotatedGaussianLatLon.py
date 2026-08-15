import numpy as np

from EarthSystemGrids.base import StructuredQuadMesh, _R_EARTH, rotate_lonlat
from EarthSystemGrids import GaussianLatLon


def generate_mesh(lat, lon, rotation_axis_longitude_deg, rotation_deg,
                  mask=None, earth_radius: float = _R_EARTH,
                  attrs=None) -> StructuredQuadMesh:
    """
    Build a Gaussian lat-lon grid whose pole is displaced off Earth's true
    axis by an arbitrary rotation.

    Builds the grid natively (unrotated -- native pole at the true
    geographic pole) via GaussianLatLon.generate_mesh, then rigidly rotates
    every corner and centre coordinate about an axis lying in the
    equatorial plane at rotation_axis_longitude_deg, by rotation_deg
    (right-hand rule; see rotate_lonlat).

    Solid angle -- and therefore m² area -- is invariant under a rigid
    rotation of the sphere, so each cell's area from the unrotated mesh is
    reused unchanged rather than recomputed from the rotated corners.

    A rotation angle (true east to the grid's local i-direction) is not
    computed here: StructuredQuadMesh.rotation_angle() already derives it,
    exactly, from the rotated mesh's own corner geometry, and
    write_to_SCRIP_grid_file already includes it as grid_angle /
    grid_cos_angle / grid_sin_angle automatically.

    mask defaults to all-ocean, same convention as
    GaussianLatLon.generate_mesh: land/sea is not rotation-invariant (it
    depends on each cell's absolute, post-rotation location), so for a real
    coastline mask call EarthSystemGrids.base.apply_ocean_mask on the
    returned (already-rotated) mesh -- not before rotating.

    Parameters
    ----------
    lat, lon : as GaussianLatLon.generate_mesh -- int count or explicit
        centre array, in the grid's own native (unrotated) frame.
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
    StructuredQuadMesh, shape (nlat, nlon)
    """
    base_mesh = GaussianLatLon.generate_mesh(lat, lon, earth_radius=earth_radius)
    nlat, nlon = base_mesh.shape

    corner_lon = base_mesh.node_lon[base_mesh.face_nodes].reshape(nlat, nlon, 4)
    corner_lat = base_mesh.node_lat[base_mesh.face_nodes].reshape(nlat, nlon, 4)
    face_lon   = base_mesh.face_lon.reshape(nlat, nlon)
    face_lat   = base_mesh.face_lat.reshape(nlat, nlon)

    corner_lon, corner_lat = rotate_lonlat(
        corner_lon, corner_lat, rotation_axis_longitude_deg, rotation_deg)
    face_lon, face_lat = rotate_lonlat(
        face_lon, face_lat, rotation_axis_longitude_deg, rotation_deg)

    if mask is None:
        mask = np.ones((nlat, nlon), dtype=np.int32)

    return StructuredQuadMesh.from_corners(
        corner_lon = corner_lon,
        corner_lat = corner_lat,
        face_lon   = face_lon,
        face_lat   = face_lat,
        area       = base_mesh.area.reshape(nlat, nlon),
        mask       = np.asarray(mask),
        shape      = (nlat, nlon),
        attrs      = attrs or {},
    )
