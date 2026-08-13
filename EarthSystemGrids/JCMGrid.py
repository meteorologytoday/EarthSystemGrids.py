from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, atan2, asin, stack
from dataclasses import dataclass
from typing import List
from pathlib import Path
import xarray as xr

@dataclass
class JCMGrid:
    r_spherical: np.ndarray
    r_corners_spherical: np.ndarray
    binary_mask: np.ndarray
    grid_solid_angles: np.ndarray

def normalize(a):
    normalized = a / norm(a, ord=2, axis=0, keepdims=True)
    return normalized

def cartesian_to_spherical(r_cartesian):
    """
    Convert from Cartesian coordinate to spherical coordinate
    The 0th axis is (x, y, z), the 1st axis is the points
    """
    d = norm(r_cartesian, ord=2, axis=0)
    lat = asin(r_cartesian[2] / d)
    lon = atan2(r_cartesian[1], r_cartesian[0])
    return stack((d, lon, lat), axis=0)

def spherical_to_cartesian(r_sphere):
    """
    Convert from spherical coordinate to lon lat
    The 0th axis is (r, lon, lat), the 1st axis is the points
    """
    x = r_sphere[0] * cos(r_sphere[2]) * cos(r_sphere[1])
    y = r_sphere[0] * cos(r_sphere[2]) * sin(r_sphere[1])
    z = r_sphere[0] * sin(r_sphere[2])
    return stack((x, y, z), axis=0)

def compute_solid_angle(r_corners_spherical):

    r_corners_cartesian = spherical_to_cartesian(r_corners_spherical)
    vec1 = (  ( r_corners_cartesian[:, 1, :, :] - r_corners_cartesian[:, 0, :, :] ) 
            + ( r_corners_cartesian[:, 2, :, :] - r_corners_cartesian[:, 3, :, :] ) ) / 2 
    vec2 = (  ( r_corners_cartesian[:, 2, :, :] - r_corners_cartesian[:, 1, :, :] ) 
            + ( r_corners_cartesian[:, 3, :, :] - r_corners_cartesian[:, 0, :, :] ) ) / 2 

    # outer product in-place
    d0 =   vec1[1] * vec2[2] - vec1[2] * vec2[1]
    d1 = - vec1[0] * vec2[2] + vec1[2] * vec2[0]
    d2 =   vec1[0] * vec2[1] - vec1[1] * vec2[0]
    areas = (d0**2 + d1**2 + d2**2)**0.5
    solid_angles = areas / np.sum(areas) * np.pi * 4

    return solid_angles

def _horizontal_grid_for_resolution(resolution: int):
    """
    The dinosaur.spherical_harmonic.Grid for a JCM spectral truncation.

    Mirrors jcm.utils.get_coords's own horizontal-grid selection, without
    the vertical-coordinate / physics machinery that function also builds:
    JCMGrid only needs the horizontal grid's node coordinates and
    quadrature weights, not a full model CoordinateSystem. T63 has no
    dedicated dinosaur.spherical_harmonic.Grid.T63 factory, so it is built
    directly via Grid.construct with the same (max_wavenumber=63,
    gaussian_nodes=48) pairing jcm.utils.get_coords uses, which reproduces
    the standard 192 x 96 ECHAM T63 grid.
    """
    import dinosaur.spherical_harmonic as dsh
    from jcm.utils import VALID_TRUNCATIONS

    if resolution not in VALID_TRUNCATIONS:
        raise ValueError(
            f"Invalid resolution (spectral truncation) {resolution}. "
            f"Must be one of: {VALID_TRUNCATIONS}."
        )
    if resolution == 63:
        return dsh.Grid.construct(max_wavenumber=63, gaussian_nodes=48)
    return getattr(dsh.Grid, f"T{resolution}")()


def _latitude_bounds(lat_centers):
    """
    Cell-edge latitudes bisecting between consecutive Gaussian latitude
    centres, with the two outer edges clamped to the poles.

    Gaussian quadrature latitudes are not evenly spaced (they compress
    towards the poles), so there is no single dlat the way there is for
    the model's uniformly-spaced longitudes. A naive symmetric
    extrapolation of the outermost centres falls noticeably short of the
    true pole (~1 deg short at T31) -- consistent with the model's own
    grid.quadrature_weights, whose exact sum over the whole grid is 4*pi,
    the outermost band's cell extends all the way to +/-pi/2.
    """
    midpoints = 0.5 * (lat_centers[1:] + lat_centers[:-1])
    return np.concatenate([[-np.pi/2], midpoints, [np.pi/2]])


def generate_JCMGrid(
    resolution: int,
):
    horizontal_grid = _horizontal_grid_for_resolution(resolution)

    # jcm/dinosaur's own native grid: longitude is uniformly spaced and
    # spans the full periodic 360 deg (so the same linspace-of-bounds
    # construction the old terrain-file-based code used is still exact
    # here); latitude is the non-uniform Gaussian quadrature grid.
    lon_centers = np.asarray(horizontal_grid.longitudes)
    lat_centers = np.asarray(horizontal_grid.latitudes)
    nlon, nlat  = horizontal_grid.nodal_shape

    dlon = lon_centers[1] - lon_centers[0]
    lon_bounds = np.linspace(lon_centers[0] - dlon/2, lon_centers[-1] + dlon/2, nlon+1)
    lat_bounds = _latitude_bounds(lat_centers)

    # JCM is lon-lat
    r_spherical = np.zeros((3, nlon, nlat))
    r_corners_spherical = np.zeros((3, 4, nlon, nlat))
    for i in range(nlon):
        for j in range(nlat):
            r_spherical[:, i, j] = [1.0, lon_centers[i], lat_centers[j]]
            r_corners_spherical[:, 0, i, j] = [1.0, lon_bounds[i], lat_bounds[j]]
            r_corners_spherical[:, 1, i, j] = [1.0, lon_bounds[i+1], lat_bounds[j]]
            r_corners_spherical[:, 2, i, j] = [1.0, lon_bounds[i+1], lat_bounds[j+1]]
            r_corners_spherical[:, 3, i, j] = [1.0, lon_bounds[i], lat_bounds[j+1]]

    # Construct land-sea mask (placeholder: all ocean, no coastline data wired in)
    binary_mask = np.ones((nlon, nlat))

    # Construct solid angles
    grid_solid_angles = compute_solid_angle(r_corners_spherical)

    return JCMGrid(
        r_spherical = r_spherical,
        r_corners_spherical = r_corners_spherical,
        binary_mask = binary_mask,
        grid_solid_angles = grid_solid_angles,
    )

    
def write_to_SCRIP_grid_file(grid: JCMGrid, output_file: str | Path, flatten:bool = True):
    
    grid_size = grid.binary_mask.size
    grid_corners = 4
    grid_dims = list(grid.binary_mask.shape)
    
    # After testing, I found the order assumed in ESGM_RegridWeightGen is reversed.
    # This is undocumented in their user manual
    grid_dims = grid_dims[::-1] 
    
    grid_dim_names = ["lon", "lat"]
    grid_center_lon = grid.r_spherical[1]
    grid_center_lat = grid.r_spherical[2]
    grid_imask = grid.binary_mask
    grid_area = grid.grid_solid_angles
   
    grid_corner_lon = np.permute_dims( grid.r_corners_spherical[1], axes=(1, 2, 0))
    grid_corner_lat = np.permute_dims( grid.r_corners_spherical[2], axes=(1, 2, 0))

    rad2deg = 180/np.pi
    if flatten:
        ds = xr.Dataset(
            data_vars = dict(
                grid_dims = ( ["grid_rank", ], grid_dims),
                grid_imask = ( ["grid_size", ], grid_imask.flatten()),
                grid_center_lat = ( ["grid_size", ], grid_center_lat.flatten() * rad2deg, {"units" : "degrees"} ),
                grid_center_lon = ( ["grid_size", ], grid_center_lon.flatten() * rad2deg, {"units" : "degrees"} ),
                grid_corner_lat = ( ["grid_size", "grid_corners"], grid_corner_lat.reshape((-1, grid_corners)) * rad2deg, {"units" : "degrees"} ),
                grid_corner_lon = ( ["grid_size", "grid_corners"], grid_corner_lon.reshape((-1, grid_corners)) * rad2deg, {"units" : "degrees"} ),
                grid_area = ( ["grid_size",], grid_area.flatten(), {"units" : "radians^2"} ),
            ),
        )

    else:
 
        ds = xr.Dataset(
            data_vars = dict(
                grid_dims = ( ["grid_rank", ], grid_dims),
                grid_imask = ( [*grid_dim_names], grid_imask),
                grid_center_lat = ( [*grid_dim_names], grid_center_lat, {"units" : "radians"} ),
                grid_center_lon = ( [*grid_dim_names], grid_center_lon, {"units" : "radians"} ),
                grid_corner_lat = ( [*grid_dim_names, "grid_corners"], grid_corner_lat, {"units" : "radians"} ),
                grid_corner_lon = ( [*grid_dim_names, "grid_corners"], grid_corner_lon, {"units" : "radians"} ),
                grid_area = ( [*grid_dim_names], grid_area, {"units" : "radians^2"} ),
            ),
        )

    ds.to_netcdf(output_file)


def test_output_SCRIP_file():
    output_file = "grid_JCM.nc"

    print("Generating grid...") 
    grid = generate_JCMGrid(31)

    print("Writing to file: ", output_file)
    write_to_SCRIP_grid_file(grid, output_file, flatten=False)

if __name__ == "__main__":
    test_output_SCRIP_file()
     
