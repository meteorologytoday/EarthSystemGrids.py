import numpy as np


def rotate_lonlat(lon, lat, axis_longitude_deg, rotation_deg):
    """
    Rigidly rotate points on the unit sphere about an arbitrary axis.

    The rotation axis lies in the equatorial plane, at longitude
    axis_longitude_deg; rotation_deg is the rotation angle about that axis,
    right-hand rule (thumb pointing from Earth's centre out through the
    axis longitude on the equator). This is the parameterization a
    "rotated pole" grid needs: rotation_deg controls how far the native
    pole is displaced from the true pole, axis_longitude_deg controls which
    direction it's displaced towards.

    Uses Rodrigues' rotation formula on the points' Cartesian unit vectors
    rather than an explicit orthonormal frame, so it stays numerically
    stable even when a point sits exactly on the rotation axis (see
    StructuredQuadMesh.rotation_angle, which documents the same tradeoff
    for a different formula) -- there the in-plane component is
    legitimately zero, and Rodrigues returns the point unchanged instead of
    dividing by a near-zero norm.

    Parameters
    ----------
    lon, lat : array-like, radians, any broadcast-compatible shape
    axis_longitude_deg, rotation_deg : float, degrees

    Returns
    -------
    lon_rotated, lat_rotated : radians, broadcast(lon, lat).shape
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    axis_lon = np.deg2rad(axis_longitude_deg)
    angle    = np.deg2rad(rotation_deg)
    axis     = np.array([np.cos(axis_lon), np.sin(axis_lon), 0.0])

    p = np.stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ], axis=0)   # (3, *shape)

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    k_cross_p = np.stack([
        axis[1] * p[2] - axis[2] * p[1],
        axis[2] * p[0] - axis[0] * p[2],
        axis[0] * p[1] - axis[1] * p[0],
    ], axis=0)
    k_dot_p = axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2]
    axis_bc = axis.reshape((3,) + (1,) * (p.ndim - 1))

    p_rotated = (
        p * cos_angle
        + k_cross_p * sin_angle
        + axis_bc * k_dot_p * (1 - cos_angle)
    )

    lon_rotated = np.arctan2(p_rotated[1], p_rotated[0])
    lat_rotated = np.arcsin(np.clip(p_rotated[2], -1.0, 1.0))
    return lon_rotated, lat_rotated
