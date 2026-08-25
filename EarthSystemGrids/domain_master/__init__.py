"""
DomainMaster unifies domains used in an Earth System Model -- land, ocean,
atmosphere, and an "exchange grid" they hand fields to each other through --
each carrying its own grid, land-sea mask, and optional topography, plus
regridding transformations between domains (ESMF weight-file-backed or a
user-supplied callable) and a coverage check for catching gaps ("holes") or
double-covered cells ("overlaps") when multiple sources regrid onto a
shared exchange grid.

Split across submodules by concern (types.py, coverage.py, regrid.py,
loading.py, domain_master.py), matching EarthSystemGrids.base's
one-file-per-concern layout; this __init__ re-exports the public API so
`from EarthSystemGrids.domain_master import DomainMaster` etc. keeps
working unchanged.
"""

from EarthSystemGrids.domain_master.coverage import CoverageError, CoverageReport, check_coverage
from EarthSystemGrids.domain_master.domain_master import DomainMaster, ValidationError
from EarthSystemGrids.domain_master.types import Domain, Transformation

__all__ = [
    "Domain",
    "Transformation",
    "CoverageError",
    "CoverageReport",
    "check_coverage",
    "DomainMaster",
    "ValidationError",
]
