# Why?
When developing Earth system models, handling horizontal grids can be difficult due to the singularity of the latitude-longitude system at the pole. Therefore, cubic-sphere, LatLonCap, and many other interesting grids are created to deal with singularities. This repository aims to provide grid generation, spatial operators, and grid-remapping code (compatible with [ESMF_RegridWeightGen](https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node3.html#SECTION03020000000000000000)), saving Python Earth modelers a significant amount of time. 

This project also provides interactive Jupyter notebooks for education.

# What Grids?
1. Gaussian lat-lon
2. LatLonCap
3. CubicSphere
4. Icosahedral
5. ORCA ([tripolar grid](https://www.nemo-ocean.eu/doc/node108.html))
6. Geodesic Grids

# Grid I/O
Grids can be written to and read back from SCRIP grid files, so they can be handed directly to `ESMF_RegridWeightGen` (or tools built on it, like `ncremap`) for remapping.
