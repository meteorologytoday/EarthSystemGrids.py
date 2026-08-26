"""
Loading grids and land-sea-mask/topography fields from disk for
register_domain -- SCRIP/CF grid files, and netCDF fields.
"""

from typing import Optional

import numpy as np
import xarray as xr

from EarthSystemGrids.base.StructuredQuadMesh import StructuredQuadMesh
from EarthSystemGrids.base.UnstructuredGridMesh import UnstructuredGridMesh


def _is_mesh(obj) -> bool:
    return hasattr(obj, "face_lon") and hasattr(obj, "mask") and hasattr(obj, "area")


def _load_grid(path, grid_format: Optional[str] = None):
    """
    Load a mesh from a SCRIP or CF grid file, auto-detecting the format
    (based on what this repo's own writers emit) unless grid_format is
    given explicitly ("SCRIP" or "CF").

    Tries StructuredQuadMesh first (covers JCM/GaussianLatLon/
    RotatedGaussianLatLon/DisplacedPoleGrid output, all uniform
    quadrilateral meshes), falling back to UnstructuredGridMesh if that
    raises (a non-quadrilateral mesh -- e.g. a future arbitrary-polygon
    grid).
    """
    if grid_format is None:
        with xr.open_dataset(path) as ds:
            if "grid_dims" in ds.variables and "grid_center_lon" in ds.variables:
                grid_format = "SCRIP"
            elif "mesh" in ds.variables and ds["mesh"].attrs.get("cf_role") == "mesh_topology":
                grid_format = "CF"
            else:
                raise ValueError(
                    f"could not auto-detect grid format for {path!r} -- "
                    "pass grid_format='SCRIP' or 'CF' explicitly."
                )

    loader_name = {"SCRIP": "from_SCRIP_file", "CF": "from_CF_file"}.get(grid_format)
    if loader_name is None:
        raise ValueError(f"grid_format must be 'SCRIP' or 'CF', got {grid_format!r}")

    try:
        return getattr(StructuredQuadMesh, loader_name)(path)
    except ValueError:
        return getattr(UnstructuredGridMesh, loader_name)(path)


def _load_field(path, var_name: str) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        return ds[var_name].squeeze().values.reshape(-1)
