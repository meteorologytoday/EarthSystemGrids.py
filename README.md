# Why?
When developing Earth system models, handling horizontal grids can be difficult due to the singularity of the latitude-longitude system at the pole. Therefore, cubic-sphere, LatLonCap, and many other interesting grids are created to deal with singularities. This repository aims to provide grid generation, spatial operators, and grid-remapping code (compatible with [ESMF_RegridWeightGen](https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node3.html#SECTION03020000000000000000)), saving Python Earth modelers a significant amount of time. 

# Grids Available

1. Gaussian (lat-lon) grid
2. Rotated Gaussian grid
3. JCM grid

# Quickstart

To generate land-sea mask, topography, grid of JCM grid and rotated Gaussian grid, and regridding weights between them, run:
```
./script/generate_fractional_mask_via_ncremap_JCM_RGLL.sh

ls grid_data         # grid and regridding weights
ls landsea_mask_data # land-sea mask and topography
``` 


# Future Plan
1. Support a `GridManager` that integrates mask, topography, regridding of scalar and vector.
2. More grids, including
   - LatLonCap
   - CubicSphere
   - Icosahedral
   - ORCA ([tripolar grid](https://www.nemo-ocean.eu/doc/node108.html))
   - Geodesic Grids
3. Provides interactive Jupyter notebooks for education.

