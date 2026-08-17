#!/bin/bash
#
# 1. Regrid the ERA5 land-sea mask and geopotential onto the JCM grid and
#    the rotated Gaussian lat-lon grid, conservatively (area-weighted), via
#    NCO's ncremap -- both variables in the same ncremap call, so they land
#    in the same output file per grid. The geopotential is then converted
#    to topography (surface altitude) in place, by dividing by gravity.
# 2. Generate reusable ESMF regridding weight files directly between the
#    JCM grid and the rotated Gaussian lat-lon grid (no ERA5 involved), for
#    both mapping directions and both bilinear and conservative methods.
#
# ERA5's `lsm` variable is already a fractional land-sea mask (0..1) at
# 0.25deg, so conservatively remapping it directly gives a fractional mask
# on the target grid -- no thresholding step needed. ncremap infers the
# source grid from ERA5_landsea_mask.nc's own lat/lon coordinates; only the
# two destination grids need an explicit SCRIP file, generated below from
# this repo's own grid code.
#
# ERA5's `z` variable is geopotential (m2 s-2), not elevation. Conservative
# remapping is linear, so it commutes with the division by gravity -- the
# script remaps `z` first and divides afterward, rather than dividing ERA5
# `z` upfront, for exactly the same result with one fewer intermediate file.
#
# The JCM<->rotated-Gaussian weight files ignore masking entirely -- every
# cell on both grids is treated as valid. This holds by construction, not
# by an extra flag: neither grid below has apply_ocean_mask applied, so
# each SCRIP file's grid_imask is all 1s, and ESMF_RegridWeightGen simply
# has nothing to exclude.
#
# Needs ncremap (NCO) and ESMF_RegridWeightGen on PATH.

set -euo pipefail
set -x

# --- inputs you may want to change ------------------------------------------
ERA5_FILE=ERA5_landsea_mask.nc     # must have variables "lsm" and "z" on lat/lon
OUTPUT_DIR=landsea_mask_data
GRID_DIR=grid_data

GRAVITY=9.80665                    # m s-2, standard gravity (ERA5 convention: elevation = z / GRAVITY)

JCM_RESOLUTION=31                  # spectral truncation, e.g. 31 -> T31

# RotatedGaussianLatLon.generate_mesh parameters; nlat/nlon default to a
# T31-equivalent Gaussian grid (96x48) so the JCM<->rotated grid weights
# below are comparable in resolution.
RGLL_NLAT=48
RGLL_NLON=96
RGLL_ROTATION_AXIS_LONGITUDE_DEG=48.0
RGLL_ROTATION_DEG=12.0

REMAP_ALGORITHM=conserve           # area-conservative: right choice for a fractional field

WEIGHT_METHODS=(bilinear conserve) # ESMF_RegridWeightGen -m values for the JCM<->RGLL weight files
# -----------------------------------------------------------------------------

if [ ! -f "$ERA5_FILE" ]; then
    echo "error: $ERA5_FILE not found" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$GRID_DIR"

jcm_scrip="${GRID_DIR}/JCM_T${JCM_RESOLUTION}.SCRIP.nc"
rgll_scrip="${GRID_DIR}/RotatedGaussianLatLon.SCRIP.nc"

echo "Generating JCM T${JCM_RESOLUTION} SCRIP grid ..."
python3 -c "
from EarthSystemGrids.JCMGrid import generate_JCMGrid
mesh = generate_JCMGrid(${JCM_RESOLUTION})
mesh.write_to_SCRIP_grid_file('${jcm_scrip}', flatten=True)
"

echo "Generating rotated Gaussian lat-lon grid SCRIP file ..."
python3 -c "
from EarthSystemGrids.GaussianLatLon import gaussian_latitude_bounds, equally_spaced_bounds
from EarthSystemGrids.RotatedGaussianLatLon import RotatedGaussianLatLon
mesh = RotatedGaussianLatLon.generate_mesh(
    lat=gaussian_latitude_bounds(${RGLL_NLAT}),
    lon=equally_spaced_bounds(${RGLL_NLON}, 0.0, 360.0),
    rotation_axis_longitude_deg=${RGLL_ROTATION_AXIS_LONGITUDE_DEG},
    rotation_deg=${RGLL_ROTATION_DEG},
)
mesh.write_to_SCRIP_grid_file('${rgll_scrip}', flatten=True)
"

landsea_jcm="${OUTPUT_DIR}/landsea_mask_fraction_JCM_T${JCM_RESOLUTION}.nc"
landsea_rgll="${OUTPUT_DIR}/landsea_mask_fraction_RotatedGaussianLatLon.nc"

echo "Remapping ERA5 land-sea mask and geopotential onto the JCM grid ..."
ncremap -a "${REMAP_ALGORITHM}" \
    -g "${jcm_scrip}" \
    -i "${ERA5_FILE}" \
    -v lsm,z \
    -o "${landsea_jcm}"

echo "Remapping ERA5 land-sea mask and geopotential onto the rotated Gaussian lat-lon grid ..."
ncremap -a "${REMAP_ALGORITHM}" \
    -g "${rgll_scrip}" \
    -i "${ERA5_FILE}" \
    -v lsm,z \
    -o "${landsea_rgll}"

echo "Converting remapped geopotential to topography (z / gravity) ..."
python3 -c "
import xarray as xr

GRAVITY = ${GRAVITY}

for path in ['${landsea_jcm}', '${landsea_rgll}']:
    # Load fully and close before overwriting -- xarray's netCDF4 backend
    # keeps the file open lazily otherwise, and writing back to a path it
    # still has open corrupts the file.
    ds = xr.open_dataset(path).load()
    ds.close()

    topography = ds['z'] / GRAVITY
    topography.attrs = {
        'standard_name': 'surface_altitude',
        'long_name': 'surface altitude (ERA5 geopotential / gravity)',
        'units': 'm',
    }
    ds = ds.drop_vars('z')
    ds['topography'] = topography
    ds.to_netcdf(path)
"

echo "Done. Fractional land-sea mask and topography written to ${OUTPUT_DIR}/"

echo "Generating JCM <-> rotated Gaussian lat-lon grid regridding weights ..."
for method in "${WEIGHT_METHODS[@]}"; do
    ESMF_RegridWeightGen \
        -s "${jcm_scrip}" -d "${rgll_scrip}" -m "${method}" \
        -w "${GRID_DIR}/weight_algo-${method}_JCM_T${JCM_RESOLUTION}_to_RotatedGaussianLatLon.nc"
    ESMF_RegridWeightGen \
        -s "${rgll_scrip}" -d "${jcm_scrip}" -m "${method}" \
        -w "${GRID_DIR}/weight_algo-${method}_RotatedGaussianLatLon_to_JCM_T${JCM_RESOLUTION}.nc"
done

echo "Done. Regridding weights written to ${GRID_DIR}/"
