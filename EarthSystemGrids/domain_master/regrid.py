"""
Builds a flat-(nface,)-array regridder callable from a pre-generated ESMF
weight file.
"""

from typing import Callable

import xarray as xr


def _make_esmf_regridder(weight_file, *, frac_b_threshold: float = 0.0) -> Callable:
    """
    Build a flat-(nface,)-array regridder callable from an ESMF weight
    file, backed by EarthSystemGrids.base.esmf_regrid.ESMFRegridder.

    Applies the weight file's row/col/S directly to the flat (n_a,) array
    this repo's meshes already use (Domain.grid.face_lon/mask/area/etc.),
    bypassing ESMFRegridder's own 2D (n_lon, n_lat) __call__/apply_batched
    -- that reshape targets JEM's SlabGrid convention and isn't needed
    here, since this repo's own SCRIP writer already lays out its flat
    nface array in the same fastest-axis-first order ESMF expects (see
    StructuredQuadMesh.write_to_SCRIP_grid_file's grid_dims=[ni,nj]).

    frac_b isn't exposed on ESMFWeights, so it's read separately here.
    Destination cells with zero source coverage (frac_b <= frac_b_threshold)
    come back NaN, so check_coverage can see them as holes; partial
    coverage (0 < frac_b < 1) is kept as a legitimate area-weighted value.

    jax is imported lazily here (not at module top), so DomainMaster and
    this module import cleanly without jax installed as long as only
    custom (non-weight_file) regridders are actually used.
    """
    from EarthSystemGrids.base.esmf_regrid import ESMFRegridder
    import jax.numpy as jnp

    r = ESMFRegridder(weight_file)
    w = r.weights
    with xr.open_dataset(weight_file) as ds:
        frac_b = jnp.asarray(ds["frac_b"].values)

    def regridder(field_a):
        field_a = jnp.asarray(field_a)
        dst = jnp.zeros(w.dst_size, dtype=field_a.dtype).at[w.row_indices].add(
            field_a[w.col_indices] * w.weights
        )
        return jnp.where(frac_b <= frac_b_threshold, jnp.nan, dst)

    return regridder
