#!/bin/bash
#
# Regrid the ERA5 land-sea mask onto the JCM grid and the displaced-pole
# grid, conservatively (area-weighted), via NCO's ncremap.
#
# ERA5's `lsm` variable is already a fractional land-sea mask (0..1) at
# 0.25deg, so conservatively remapping it directly gives a fractional mask
# on the target grid -- no thresholding step needed. ncremap infers the
# source grid from ERA5_landsea_mask.nc's own lat/lon coordinates; only the
# two destination grids need an explicit SCRIP file, generated below from
# this repo's own grid code.
#
# Needs ncremap (NCO) and ESMF_RegridWeightGen on PATH.

set -euo pipefail
set -x

# --- inputs you may want to change ------------------------------------------
ERA5_FILE=ERA5_landsea_mask.nc     # must have variable "lsm" on lat/lon
OUTPUT_DIR=landsea_mask_data
GRID_DIR=grid_data

JCM_RESOLUTION=31                  # spectral truncation, e.g. 31 -> T31

# DisplacedPoleGrid.build_example_grid() defaults; override any of these
DPG_NUMBER_OF_ROWS_NH=30
DPG_NUMBER_OF_COLUMNS=120
DPG_DLAT_IN_SH_DEGREE=3.0
DPG_FORMULATION=mi1996

REMAP_ALGORITHM=conserve           # area-conservative: right choice for a fractional field
# -----------------------------------------------------------------------------

if [ ! -f "$ERA5_FILE" ]; then
    echo "error: $ERA5_FILE not found" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$GRID_DIR"

jcm_scrip="${GRID_DIR}/JCM_T${JCM_RESOLUTION}.SCRIP.nc"
dpg_scrip="${GRID_DIR}/DisplacedPoleGrid.SCRIP.nc"

echo "Generating JCM T${JCM_RESOLUTION} SCRIP grid ..."
python3 -c "
from EarthSystemGrids.JCMGrid import generate_JCMGrid
mesh = generate_JCMGrid(${JCM_RESOLUTION})
mesh.write_to_SCRIP_grid_file('${jcm_scrip}', flatten=True)
"

echo "Generating displaced-pole grid SCRIP file ..."
python3 -c "
from EarthSystemGrids.DisplacedPoleGrid import build_example_grid
dpg = build_example_grid(
    number_of_rows_in_NH=${DPG_NUMBER_OF_ROWS_NH},
    number_of_columns=${DPG_NUMBER_OF_COLUMNS},
    dlat_in_SH_degree=${DPG_DLAT_IN_SH_DEGREE},
    formulation='${DPG_FORMULATION}',
)
mesh = dpg.generate_mesh()
mesh.write_to_SCRIP_grid_file('${dpg_scrip}', flatten=True)
"

echo "Remapping ERA5 land-sea mask onto the JCM grid ..."
ncremap -a "${REMAP_ALGORITHM}" \
    -g "${jcm_scrip}" \
    -i "${ERA5_FILE}" \
    -v lsm \
    -o "${OUTPUT_DIR}/landsea_mask_fraction_JCM_T${JCM_RESOLUTION}.nc"

echo "Remapping ERA5 land-sea mask onto the displaced-pole grid ..."
ncremap -a "${REMAP_ALGORITHM}" \
    -g "${dpg_scrip}" \
    -i "${ERA5_FILE}" \
    -v lsm \
    -o "${OUTPUT_DIR}/landsea_mask_fraction_DisplacedPoleGrid.nc"

echo "Done. Fractional masks written to ${OUTPUT_DIR}/"
