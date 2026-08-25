"""
Plain data containers for DomainMaster's registry: a Domain (grid + masks)
and a Transformation (regridding method between two domains).
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

from EarthSystemGrids.base.UnstructuredGridMesh import UnstructuredGridMesh


@dataclass
class Domain:
    name: str
    grid: Optional[UnstructuredGridMesh] = None   # None -- a placeholder, name declared but not yet resolved
    land_fraction: Optional[np.ndarray] = None  # fractional (e.g. ocean fraction 0..1); always None while grid is None
    mask: Optional[np.ndarray] = None              # 1 = active, 0 = inactive; independent of land_fraction above
    topography: Optional[np.ndarray] = None
    is_exchange_grid: bool = False
    attrs: dict = field(default_factory=dict)


@dataclass
class Transformation:
    source: str
    target: str
    method: str
    weight_file: Optional[Union[str, os.PathLike]] = None
    regridder: Optional[Callable[[Any], Any]] = None   # resolved callable; built lazily for weight_file, used as-is if given directly
