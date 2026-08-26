"""
Coverage checking: verifying that a set of per-source fields, each already
regridded onto a shared destination grid, together cover it with no gaps
("holes") and no ambiguous double-coverage ("overlaps").
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


class CoverageError(Exception):
    """
    Raised by check_coverage when a set of per-source contributions
    regridded onto an exchange grid leaves holes (uncovered valid cells)
    or, if allow_overlap=False, overlaps (a valid cell covered by more
    than one source).
    """


@dataclass
class CoverageReport:
    valid_mask: np.ndarray             # (n_b,) bool -- cells that needed coverage
    covered_by: Dict[str, np.ndarray]  # {source_name: (n_b,) bool}
    holes: np.ndarray                  # valid & covered by nobody
    overlaps: np.ndarray               # valid & covered by >1 source
    ok: bool                           # no holes, and (allow_overlap or no overlaps)


def _format_indices(mask: np.ndarray, limit: int = 20) -> str:
    idx = np.flatnonzero(mask)
    shown = idx[:limit].tolist()
    suffix = f", ... ({idx.size} total)" if idx.size > limit else ""
    return f"{shown}{suffix}"


def check_coverage(
    valid_mask,
    contributions: Dict[str, Any],
    *,
    allow_overlap: bool = False,
    raise_error: bool = True,
) -> CoverageReport:
    """
    Check that a set of per-source fields, each already regridded onto the
    same destination grid (e.g. via DomainMaster.transform_scalar/
    transform_vector), together cover every cell `valid_mask` marks as
    needing coverage -- with no gaps ("holes") and, unless
    allow_overlap=True, no cell claimed by more than one source
    ("overlaps"). This is a plain, stateless function: a single transform
    call from one source can't see other sources' contributions, so the
    caller regrids each source independently and collects the results here.

    A contribution's NaN entries are treated as "not covered by this
    source" -- the convention DomainMaster's own ESMF-backed regridder
    follows (via each weight file's frac_b); a custom regridder is
    responsible for its own NaN convention if it wants to participate here.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    covered_by = {
        name: valid_mask & ~np.isnan(np.asarray(arr, dtype=float))
        for name, arr in contributions.items()
    }
    if covered_by:
        union_covered = np.logical_or.reduce(list(covered_by.values()))
        overlap_count = np.sum([c.astype(int) for c in covered_by.values()], axis=0)
    else:
        union_covered = np.zeros_like(valid_mask)
        overlap_count = np.zeros(valid_mask.shape, dtype=int)

    holes = valid_mask & ~union_covered
    overlaps = valid_mask & (overlap_count > 1)
    ok = not holes.any() and (allow_overlap or not overlaps.any())

    report = CoverageReport(
        valid_mask=valid_mask, covered_by=covered_by,
        holes=holes, overlaps=overlaps, ok=ok,
    )
    if raise_error and not ok:
        parts = []
        if holes.any():
            parts.append(
                f"{int(holes.sum())} hole cell(s) covered by no source: {_format_indices(holes)}"
            )
        if overlaps.any() and not allow_overlap:
            parts.append(
                f"{int(overlaps.sum())} overlapping cell(s) covered by more than one "
                f"source: {_format_indices(overlaps)}"
            )
        raise CoverageError("; ".join(parts))
    return report
