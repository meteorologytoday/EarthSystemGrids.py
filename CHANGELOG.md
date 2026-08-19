# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-08-19

Initial beta release. Establishes a tested grid I/O core and ships a small,
fully-supported set of grid types rather than every grid on the long-term
roadmap.

### Added
- Shared mesh base layer (`base/StructuredQuadMesh`, `base/UnstructuredGridMesh`)
  with UGRID-native storage, a common CF grid writer, and SCRIP round-trip
  I/O (`to_SCRIP_file` / `from_SCRIP_file`) shared by every grid type.
- Four fully-supported grid types, each with SCRIP + CF/UGRID round-trip
  tests:
  - `GaussianLatLon`
  - `RotatedGaussianLatLon`
  - `JCMGrid`
  - `DisplacedPoleGrid`
- `grid_angle` / `grid_cos_angle` / `grid_sin_angle` fields written to
  SCRIP/CF output, documenting the vector rotation formula for later use by
  spatial operators.
- `native_lat_bounds` / `native_lon_bounds` support for `RotatedGaussianLatLon`.
- ERA5-derived land-sea mask and topography generation for the JCM and
  rotated Gaussian lat-lon grids, plus JCM \<-> displaced-pole regridding
  weight generation, driven by `script/generate_fractional_mask_via_ncremap_JCM_RGLL.sh`
  and `script/generate_fractional_mask_via_ncremap_JCM_DSP.sh`.
- `ERA5_landsea_mask.nc` reference file, tracked under `EarthSystemGrids/data/`.

### Changed
- Consolidated grid construction onto the shared `StructuredQuadMesh` /
  `UnstructuredGridMesh` base classes, replacing per-grid isinstance checks
  and deferred imports with an `extra_variables()` hook.
- `GaussianLatLon.generate_mesh` now constructs grids from cell faces rather
  than cell centres.
- README rewritten to describe only the beta-supported grids (Gaussian
  lat-lon, rotated Gaussian lat-lon, JCM, displaced-pole); `CubeSphere` and
  `LatLonCap` are documented as unimplemented, planned grids rather than
  shipped features.
- Regridding helper scripts moved from the repo root into `script/`.

### Known limitations
- `CubeSphere` and `LatLonCap` are present in the source tree but are not
  tested or supported in this release.
- Spatial operators (gradient, divergence, interpolation, vector rotation)
  are not implemented yet; the grid metadata needed for them is written to
  SCRIP/CF output but unused.
