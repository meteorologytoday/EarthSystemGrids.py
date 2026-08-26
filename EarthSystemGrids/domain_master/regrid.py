"""
Regridder-callable builders: a flat-(nface,)-array regridder from a
pre-generated ESMF weight file, and a same-grid masking regridder for
transformations that are really just a mask, not an interpolation.
"""

from typing import Callable

import numpy as np
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


def make_mask_regridder(mask, fill_value: float = np.nan) -> Callable:
    """
    Build a same-grid "masking" regridder: field_a -> where(mask, field_a, fill_value).

    For transformations where source and target share the same underlying
    grid (e.g. splitting a shared exchange grid into separate ocean/land
    views) -- no interpolation, just keeping the cells `mask` selects and
    replacing the rest with `fill_value`. `mask` is typically a domain's
    own `mask` field (1 = active, 0 = inactive; or any 0/1-or-bool array
    the same size as the grid); truthy entries are kept, falsy entries
    become `fill_value`.

    `mask` may also be a zero-argument callable returning the mask array,
    e.g. `lambda: dm.domains["ocn"].mask` -- resolved lazily, on the first
    call to the returned regridder rather than at make_mask_regridder()
    call time, and cached from then on. This lets a masking transformation
    be registered (`register_transformation(..., regridder=make_mask_regridder(
    lambda: dm.domains["ocn"].mask))`) before the domain it reads the mask
    from has one set yet -- the same "declare now, resolve later" pattern
    DomainMaster's own placeholder domains/transformations already follow.
    A callable that still returns None (or raises, e.g. because the domain
    isn't registered yet) only surfaces as an error at that first real
    call -- transform_scalar/transform_vector, or validate()'s functional
    check, not here.

    `fill_value` defaults to NaN, matching the "not covered by this
    source" convention check_coverage already reads (via each ESMF weight
    file's frac_b) -- so a set of masking transformations built this way
    (e.g. one per land/ocean view of a shared exchange grid) can be
    checked directly with check_coverage to confirm the masks partition
    the shared grid exactly, with no gaps or overlap.

    Plain numpy, not jax -- unlike the ESMF weight-file regridder above,
    nothing here runs inside a jax.jit/grad trace, so this keeps the
    implementation simple rather than tracking jax array types through a
    masking op that's normally cheap and eager.

    Raises ValueError if a concrete (non-callable) `mask` is None -- e.g.
    a domain's own `mask` field that was never set, passed directly rather
    than via a lazy callable. Silently accepting None would otherwise turn
    into a 0-d False mask that NaNs out every cell, which is exactly the
    kind of silent, all-cells-empty bug this should fail loudly on
    instead. A callable is exempt from this eager check by design -- that
    is exactly the deferred-resolution case above.
    """
    if not callable(mask) and mask is None:
        raise ValueError("make_mask_regridder needs a mask, got None")

    resolved = {}

    def _resolved_mask():
        if "value" not in resolved:
            m = mask() if callable(mask) else mask
            if m is None:
                raise ValueError("make_mask_regridder's mask callable returned None")
            resolved["value"] = np.asarray(m).astype(bool)
        return resolved["value"]

    if not callable(mask):
        _resolved_mask()  # eager: resolve (and cache) immediately, as before

    def regridder(field_a):
        return np.where(_resolved_mask(), np.asarray(field_a), fill_value)

    return regridder
