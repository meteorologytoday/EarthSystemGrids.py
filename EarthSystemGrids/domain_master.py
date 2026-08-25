"""
DomainMaster unifies domains used in an Earth System Model -- land, ocean,
atmosphere, and an "exchange grid" they hand fields to each other through --
each carrying its own grid, land-sea mask, and optional topography, plus
regridding transformations between domains (ESMF weight-file-backed or a
user-supplied callable) and a coverage check for catching gaps ("holes") or
double-covered cells ("overlaps") when multiple sources regrid onto a
shared exchange grid.
"""

import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import xarray as xr

from EarthSystemGrids.base.StructuredQuadMesh import StructuredQuadMesh
from EarthSystemGrids.base.UnstructuredGridMesh import UnstructuredGridMesh


@dataclass
class Domain:
    name: str
    grid: Optional[UnstructuredGridMesh] = None   # None -- a placeholder, name declared but not yet resolved
    landsea_mask: Optional[np.ndarray] = None      # always None while grid is None (no face count to default/validate against)
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


class CoverageError(Exception):
    """
    Raised by check_coverage when a set of per-source contributions
    regridded onto an exchange grid leaves holes (uncovered valid cells)
    or, if allow_overlap=False, overlaps (a valid cell covered by more
    than one source).
    """


class ValidationError(Exception):
    """
    Raised by DomainMaster.validate() when the registered domains and
    transformations aren't internally consistent -- a transformation
    referencing a domain that was never registered, or one whose regridder
    doesn't actually map source-sized fields to target-sized ones.
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


def _angle_trig(mesh) -> Tuple[np.ndarray, np.ndarray]:
    return mesh.extra_variables["cos_angle"].values, mesh.extra_variables["sin_angle"].values


class DomainMaster:
    """
    Registry of named domains (grid + land-sea mask + optional topography)
    and the regridding transformations between them.

    A domain that other components regrid fields onto is registered the
    same way as any other domain, just with is_exchange_grid=True --
    DomainMaster doesn't have a separate registration path for it.
    """

    def __init__(self):
        self._domains: Dict[str, Domain] = {}
        self._transformations: Dict[Tuple[str, str, str], Transformation] = {}

    @property
    def exchange_grid_names(self) -> List[str]:
        return [name for name, d in self._domains.items() if d.is_exchange_grid]

    @property
    def domains(self) -> Mapping[str, Domain]:
        """Read-only view of registered domains, keyed by name."""
        return MappingProxyType(self._domains)

    @property
    def transformations(self) -> Mapping[Tuple[str, str, str], Transformation]:
        """Read-only view of registered transformations, keyed by (source, target, method)."""
        return MappingProxyType(self._transformations)

    def register_domain(
        self,
        name: str,
        grid=None,
        *,
        landsea_mask=None,
        topography=None,
        is_exchange_grid: bool = False,
        landsea_mask_var: str = "lsm",
        topography_var: str = "topography",
        grid_format: Optional[str] = None,
        overwrite: bool = False,
        attrs: Optional[dict] = None,
    ) -> Domain:
        """
        Register a named domain. `grid`/`landsea_mask`/`topography` may
        each be an already-constructed object (a mesh instance; an
        array-like) or a file path to load (a SCRIP/CF grid file; a
        netCDF file with a `landsea_mask_var`/`topography_var` variable).
        Passing the same path to both landsea_mask= and topography= works
        correctly -- each pulls its own named variable -- which fits the
        landsea_mask_data/ files this repo's own scripts already produce.

        `grid` may also be omitted entirely -- register_domain(name) alone
        declares a placeholder domain (just the name) to be filled in
        later by calling register_domain again with the same name and a
        grid this time (allowed without overwrite=True as long as the
        existing entry is still a placeholder; overwrite=True is required
        to replace an already-resolved domain). landsea_mask/topography
        can't be given without a grid in the same call -- there's no face
        count yet to default/validate them against. Each call fully
        specifies the domain's state; filling in a placeholder later
        doesn't carry over is_exchange_grid/attrs from the placeholder
        call -- repeat them if still wanted.
        """
        existing = self._domains.get(name)
        existing_is_resolved = existing is not None and existing.grid is not None
        if existing is not None and existing_is_resolved and not overwrite:
            raise ValueError(
                f"domain {name!r} is already registered (pass overwrite=True to replace it)"
            )

        if grid is None:
            if landsea_mask is not None or topography is not None:
                raise ValueError(
                    f"domain {name!r}: landsea_mask/topography can't be set without a grid "
                    "(there's no face count yet to validate/default them against) -- "
                    "pass grid= in this same call, or register it first"
                )
            domain = Domain(name=name, is_exchange_grid=is_exchange_grid, attrs=dict(attrs or {}))
            self._domains[name] = domain
            return domain

        if not _is_mesh(grid):
            grid = _load_grid(grid, grid_format=grid_format)
        nface = grid.face_lon.size

        def _resolve_field(value, var_name, label):
            if value is None:
                return None
            if isinstance(value, (str, os.PathLike)):
                value = _load_field(value, var_name)
            else:
                value = np.asarray(value).reshape(-1)
            if value.size != nface:
                raise ValueError(
                    f"{label} for domain {name!r} has size {value.size}, "
                    f"expected {nface} to match the grid"
                )
            return value

        landsea_mask_arr = _resolve_field(landsea_mask, landsea_mask_var, "landsea_mask")
        topography_arr = _resolve_field(topography, topography_var, "topography")

        domain = Domain(
            name=name,
            grid=grid,
            landsea_mask=landsea_mask_arr,
            topography=topography_arr,
            is_exchange_grid=is_exchange_grid,
            attrs=dict(attrs or {}),
        )
        self._domains[name] = domain
        return domain

    def register_transformation(
        self,
        source: str,
        target: str,
        method: str,
        *,
        weight_file=None,
        regridder: Optional[Callable[[Any], Any]] = None,
        overwrite: bool = False,
    ) -> Transformation:
        """
        Register a regridding transformation between two domains, labeled
        by `method` (e.g. "conserve", "bilinear"). At most one of
        `weight_file` (a pre-generated ESMF weight file -- DomainMaster
        never generates weights itself) or `regridder` (a plain callable
        field_a -> field_b, for anything ESMF weights aren't the right
        tool for) may be given -- giving both is ambiguous and rejected.
        Giving neither registers a placeholder (source/target/method
        declared, regridding mechanism still to be filled in later by
        calling register_transformation again with the same source/target/
        method and a weight_file or regridder this time -- allowed without
        overwrite=True as long as the existing entry is still a
        placeholder; overwrite=True is required to replace an already
        resolved one).

        `source`/`target` don't need to already be registered domains --
        transformations can be registered before, after, or interleaved
        with the domains they reference, so a DomainMaster can be built up
        incrementally. A still-dangling reference, or a still-unresolved
        placeholder, is only an error once something actually needs it:
        transform_scalar/transform_vector (which look the domain/regridder
        up directly), or the explicit validate(). __repr__/summary() flag
        both so they're visible without waiting for one of those.
        """
        if weight_file is not None and regridder is not None:
            raise ValueError(
                "register_transformation takes at most one of weight_file or regridder "
                "(got both) -- pass neither to register a placeholder to fill in later"
            )

        key = (source, target, method)
        existing = self._transformations.get(key)
        existing_is_resolved = existing is not None and (
            existing.weight_file is not None or existing.regridder is not None
        )
        if existing is not None and existing_is_resolved and not overwrite:
            raise ValueError(
                f"transformation {source!r} -> {target!r} (method={method!r}) is already "
                "registered (pass overwrite=True to replace it)"
            )

        transformation = Transformation(
            source=source, target=target, method=method,
            weight_file=weight_file, regridder=regridder,
        )
        self._transformations[key] = transformation
        return transformation

    def _get_regridder(self, source: str, target: str, method: str) -> Callable:
        key = (source, target, method)
        if key not in self._transformations:
            raise KeyError(
                f"no transformation registered for {source!r} -> {target!r} (method={method!r})"
            )
        transformation = self._transformations[key]
        if transformation.regridder is None:
            if transformation.weight_file is None:
                raise ValueError(
                    f"transformation {source!r} -> {target!r} (method={method!r}) is still "
                    "a placeholder -- call register_transformation again with the same "
                    "source/target/method and a weight_file= or regridder= to fill it in"
                )
            transformation.regridder = _make_esmf_regridder(transformation.weight_file)
        return transformation.regridder

    def transform_scalar(self, source: str, target: str, method: str, field):
        """Regrid a scalar field (n_a,) from `source` onto `target` (n_b,)."""
        return self._get_regridder(source, target, method)(field)

    def transform_vector(self, source: str, target: str, method: str, u, v):
        """
        Regrid a vector field (u, v), defined relative to `source`'s own
        local grid i/j directions, onto `target`'s grid -- also relative
        to *its* local i/j directions.

        Algorithm: rotate (u, v) from source-grid-local to geographic
        east/north using the source mesh's cos_angle/sin_angle, regrid
        each geographic component independently through the same
        registered transformation transform_scalar would use, then rotate
        the regridded geographic components into destination-grid-local
        using the destination mesh's cos_angle/sin_angle. Geographic
        east/north is a fixed basis shared by both grids (unlike each
        mesh's own i/j frame), so these three linear steps compose
        correctly only in this order -- the standard technique for vector
        fields in ESMF/CESM-style couplers. See the formulas documented in
        EarthSystemGrids.base.StructuredQuadMesh's _VECTOR_ROTATION_COMMENT.

        Raises TypeError if either domain's grid has no well-defined local
        i-direction (e.g. an UnstructuredGridMesh built via from_polygons).
        """
        src_mesh = self._domains[source].grid
        dst_mesh = self._domains[target].grid
        for domain_name, mesh in ((source, src_mesh), (target, dst_mesh)):
            if mesh is None:
                raise TypeError(
                    f"domain {domain_name!r} has no grid set yet -- register_domain(...) "
                    "needs a grid= before transform_vector can use it"
                )
            if not hasattr(mesh, "rotation_angle"):
                raise TypeError(
                    f"transform_vector needs a StructuredQuadMesh-family grid (a "
                    f"well-defined local i-direction); domain {domain_name!r}'s grid "
                    f"({type(mesh).__name__}) has none"
                )

        cos_src, sin_src = _angle_trig(src_mesh)
        u_east = u * cos_src - v * sin_src
        v_north = u * sin_src + v * cos_src

        regridder = self._get_regridder(source, target, method)
        u_east_dst = regridder(u_east)
        v_north_dst = regridder(v_north)

        cos_dst, sin_dst = _angle_trig(dst_mesh)
        u_grid_dst = u_east_dst * cos_dst + v_north_dst * sin_dst
        v_grid_dst = -u_east_dst * sin_dst + v_north_dst * cos_dst
        return u_grid_dst, v_grid_dst

    def check_coverage(
        self,
        exchange_domain: str,
        contributions: Dict[str, Any],
        *,
        valid_mask=None,
        allow_overlap: bool = False,
        raise_error: bool = True,
    ) -> CoverageReport:
        """
        Convenience wrapper around the module-level check_coverage: resolves
        a default valid_mask from the exchange domain's own grid.mask == 1
        (the same field ESMF used as grid_imask when its weight files were
        generated) unless one is given explicitly.
        """
        domain = self._domains[exchange_domain]
        if valid_mask is None:
            if domain.grid is None:
                raise ValueError(
                    f"domain {exchange_domain!r} has no grid set yet -- either register_domain(...) "
                    "with a grid= first, or pass valid_mask= explicitly"
                )
            valid_mask = domain.grid.mask == 1
        return check_coverage(
            valid_mask, contributions, allow_overlap=allow_overlap, raise_error=raise_error
        )

    def __repr__(self) -> str:
        """
        Quick-glance dump of everything currently registered -- for
        crafting a DomainMaster incrementally: register what you have,
        print(dm) to see what's there (and what's still dangling), then
        validate()/check_coverage() once ready. For programmatic
        inspection, use the `domains`/`transformations` properties
        directly rather than parsing this text.
        """
        domains = list(self._domains.values())
        transformations = list(self._transformations.values())
        lines = [
            f"DomainMaster({len(domains)} domain(s), {len(transformations)} transformation(s))"
        ]
        for d in domains:
            tag = " [exchange]" if d.is_exchange_grid else ""
            if d.grid is None:
                lines.append(f"  domain {d.name!r}{tag}: UNRESOLVED (no grid set yet)")
                continue
            shape = getattr(d.grid, "shape", None)
            grid_desc = type(d.grid).__name__
            grid_desc += f" shape={tuple(shape)}" if shape is not None else f" nface={d.grid.face_lon.size}"
            lines.append(
                f"  domain {d.name!r}{tag}: grid={grid_desc}, "
                f"landsea_mask={'yes' if d.landsea_mask is not None else 'no'}, "
                f"topography={'yes' if d.topography is not None else 'no'}"
            )
        for t in transformations:
            if t.weight_file is not None:
                desc = f"weight_file={t.weight_file!r}"
                desc += " (built)" if t.regridder is not None else " (not built yet)"
            elif t.regridder is not None:
                desc = "custom regridder"
            else:
                desc = "UNRESOLVED (no weight_file or regridder set yet)"
            missing = []
            if t.source not in self._domains:
                missing.append(f"source {t.source!r} not registered")
            elif self._domains[t.source].grid is None:
                missing.append(f"source {t.source!r} has no grid set yet")
            if t.target not in self._domains:
                missing.append(f"target {t.target!r} not registered")
            elif self._domains[t.target].grid is None:
                missing.append(f"target {t.target!r} has no grid set yet")
            warn = f" -- MISSING: {', '.join(missing)}" if missing else ""
            lines.append(
                f"  transformation {t.source!r} -> {t.target!r} (method={t.method!r}): "
                f"{desc}{warn}"
            )
        return "\n".join(lines)

    def validate(self, *, raise_error: bool = True) -> List[str]:
        """
        Check that every registered transformation's source/target refer to
        registered domains (exactly what lazy registration defers), and
        that its regridder actually maps a source-domain-sized field to a
        target-domain-sized one -- checked functionally, by calling it on a
        zero-filled dummy field, so this works uniformly whether the
        transformation is ESMF-weight-file-backed or a custom regridder=
        callable. This also builds+caches any not-yet-built weight_file
        regridder, surfacing a malformed weight file here rather than
        silently at first real transform_scalar/transform_vector call.

        Returns the list of problems found (empty if none). Raises
        ValidationError (joining them) if raise_error=True (the default)
        and any were found. This is a broader "is my registered graph
        consistent" check than check_coverage, which is specifically about
        hole/overlap on an exchange grid.
        """
        problems: List[str] = []
        for (source, target, method), t in self._transformations.items():
            label = f"transformation {source!r} -> {target!r} (method={method!r})"
            if source not in self._domains:
                problems.append(f"{label}: source domain {source!r} is not registered")
                continue
            if target not in self._domains:
                problems.append(f"{label}: target domain {target!r} is not registered")
                continue
            if self._domains[source].grid is None:
                problems.append(f"{label}: source domain {source!r} has no grid set yet")
                continue
            if self._domains[target].grid is None:
                problems.append(f"{label}: target domain {target!r} has no grid set yet")
                continue
            if t.weight_file is None and t.regridder is None:
                problems.append(f"{label}: still a placeholder -- no weight_file or regridder set yet")
                continue

            try:
                regridder = self._get_regridder(source, target, method)
            except Exception as e:
                problems.append(f"{label}: failed to build regridder: {e}")
                continue

            src_nface = self._domains[source].grid.face_lon.size
            dst_nface = self._domains[target].grid.face_lon.size
            try:
                dummy = np.zeros(src_nface, dtype=np.float64)
                result = np.asarray(regridder(dummy))
            except Exception as e:
                problems.append(f"{label}: regridder raised {e!r} on a zero test field")
                continue
            if result.shape[-1] != dst_nface:
                problems.append(
                    f"{label}: regridder output size {result.shape[-1]} does not match "
                    f"target domain {target!r}'s size {dst_nface}"
                )

        if raise_error and problems:
            raise ValidationError("; ".join(problems))
        return problems
