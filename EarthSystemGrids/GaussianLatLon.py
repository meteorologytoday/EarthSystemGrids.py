import numpy as np

from EarthSystemGrids.base import StructuredQuadMesh, _R_EARTH


def gaussian_latitudes(nlat: int):
    """
    nlat Gaussian quadrature latitudes, degrees, ascending.

    This is the standard non-uniform latitude spacing of a spectral/
    Gaussian-grid GCM: the roots of the Legendre polynomial P_nlat, mapped
    from x = sin(lat) via arcsin. Rows compress towards the poles rather
    than being evenly spaced. The usual GCM convention is an even nlat, so
    the rows come in pairs mirrored about the equator with none sitting
    exactly on it; nlat is not required to be even here, since the
    quadrature is well-defined either way -- an odd nlat just places one
    row exactly at the equator.
    """
    x, _ = np.polynomial.legendre.leggauss(nlat)
    return np.rad2deg(np.arcsin(x))


def equally_spaced_centers(n: int, lo: float, hi: float):
    """n cell-centre values evenly spaced over [lo, hi] (same units as lo/hi)."""
    bounds = np.linspace(lo, hi, n + 1)
    return 0.5 * (bounds[:-1] + bounds[1:])


def _bounds_from_centers(centers, clamp=None):
    """
    Cell-edge values bisecting between consecutive centres, for any
    monotonically increasing centres -- evenly spaced or not, Gaussian or
    not. The two outer edges are extrapolated symmetrically from the
    outermost gap, unless `clamp` gives an explicit (lo, hi) to use
    instead -- needed for latitude, where the physical boundary is the
    pole, not whatever a symmetric extrapolation happens to land on.
    """
    midpoints = 0.5 * (centers[1:] + centers[:-1])
    if clamp is not None:
        lo, hi = clamp
    else:
        lo = centers[0] - (midpoints[0] - centers[0])
        hi = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate([[lo], midpoints, [hi]])


def generate_mesh(lat, lon, mask=None, earth_radius: float = _R_EARTH,
                  attrs=None) -> StructuredQuadMesh:
    """
    Build a lat-lon StructuredQuadMesh. Latitude and longitude are each
    independently either evenly or unevenly spaced -- "Gaussian lat-lon"
    (non-uniform latitude, uniform longitude) is the default when `lat`
    and `lon` are given as plain counts, but either axis accepts an
    explicit centre array instead, evenly spaced or not.

    Parameters
    ----------
    lat : int, or array-like of latitude CENTRES in degrees (ascending)
        An int nlat requests the standard Gaussian-grid latitudes (see
        `gaussian_latitudes`). An explicit array is used as-is -- evenly
        spaced or not, it makes no difference to how the mesh is built.
    lon : int, or array-like of longitude CENTRES in degrees (ascending)
        An int nlon requests nlon evenly-spaced centres spanning the full
        periodic 360 deg, the standard longitude spacing of every
        Gaussian-grid convention. An explicit array is used as-is.
    mask : (nlat, nlon) int, optional. 0=land 1=ocean. Defaults to all
        ocean; see EarthSystemGrids.base.apply_ocean_mask to fill this in
        from a coastline afterwards.
    earth_radius : metres, spherical Earth radius used only to convert
        each cell's solid angle into an area in m².
    attrs : dict of grid-specific metadata.

    Returns
    -------
    StructuredQuadMesh, shape (nlat, nlon)
    """
    lat_centers_deg = (gaussian_latitudes(lat) if isinstance(lat, (int, np.integer))
                       else np.asarray(lat, dtype=float))
    lon_centers_deg = (equally_spaced_centers(lon, 0.0, 360.0) if isinstance(lon, (int, np.integer))
                       else np.asarray(lon, dtype=float))

    lat_centers = np.deg2rad(lat_centers_deg)
    lon_centers = np.deg2rad(lon_centers_deg)
    nlat, nlon = lat_centers.size, lon_centers.size

    # pole-clamped: a symmetric extrapolation of the outermost gap falls
    # short of the true pole for a Gaussian grid (more so at low
    # resolution), leaving the polar cap uncovered by any cell.
    lat_bounds = _bounds_from_centers(lat_centers, clamp=(-np.pi/2, np.pi/2))
    lon_bounds = _bounds_from_centers(lon_centers)

    face_lon, face_lat = np.meshgrid(lon_centers, lat_centers)   # each (nlat, nlon)

    lon_lo = np.broadcast_to(lon_bounds[:-1], (nlat, nlon))
    lon_hi = np.broadcast_to(lon_bounds[1:],  (nlat, nlon))
    lat_lo = np.broadcast_to(lat_bounds[:-1, None], (nlat, nlon))
    lat_hi = np.broadcast_to(lat_bounds[1:,  None], (nlat, nlon))

    corner_lon = np.stack([lon_lo, lon_hi, lon_hi, lon_lo], axis=-1)   # SW SE NE NW
    corner_lat = np.stack([lat_lo, lat_lo, lat_hi, lat_hi], axis=-1)

    # A lat-lon graticule cell is bounded by exactly two meridians and two
    # latitude circles, so its solid angle has the exact closed form
    # dlon * (sin(lat_hi) - sin(lat_lo)) -- no corner-triangulation
    # approximation needed, and it telescopes to exactly 4*pi pole-to-pole
    # regardless of how the latitude rows are spaced.
    dlon    = lon_bounds[1:] - lon_bounds[:-1]
    dsinlat = np.sin(lat_bounds[1:]) - np.sin(lat_bounds[:-1])
    solid_angle = dsinlat[:, None] * dlon[None, :]
    area = solid_angle * earth_radius**2

    if mask is None:
        mask = np.ones((nlat, nlon), dtype=np.int32)

    return StructuredQuadMesh.from_corners(
        corner_lon = corner_lon,
        corner_lat = corner_lat,
        face_lon   = face_lon,
        face_lat   = face_lat,
        area       = area,
        mask       = np.asarray(mask),
        shape      = (nlat, nlon),
        attrs      = attrs or {},
    )
