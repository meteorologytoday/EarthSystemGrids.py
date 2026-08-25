import os

import numpy as np
import pytest
import scipy.sparse
import xarray as xr

from EarthSystemGrids.base.UnstructuredGridMesh import UnstructuredGridMesh
from EarthSystemGrids.domain_master import (
    CoverageError,
    DomainMaster,
    ValidationError,
    _angle_trig,
    _make_esmf_regridder,
    check_coverage,
)

# These fixtures are produced by script/generate_fractional_mask_via_ncremap_JCM_RGLL.sh
# (ncremap/ESMF_RegridWeightGen) and are gitignored (*.nc), not committed -- skip
# gracefully rather than failing when they haven't been generated locally.
_JCM_SCRIP = "grid_data/JCM_T31.SCRIP.nc"
_RGLL_SCRIP = "grid_data/RotatedGaussianLatLon.SCRIP.nc"
_JCM_TO_RGLL_CONSERVE = "grid_data/weight_algo-conserve_JCM_T31_to_RotatedGaussianLatLon.nc"
_JCM_LANDSEA = "landsea_mask_data/landsea_mask_fraction_JCM_T31.nc"
_RGLL_LANDSEA = "landsea_mask_data/landsea_mask_fraction_RotatedGaussianLatLon.nc"

_FIXTURES = [_JCM_SCRIP, _RGLL_SCRIP, _JCM_TO_RGLL_CONSERVE, _JCM_LANDSEA, _RGLL_LANDSEA]
_HAVE_FIXTURES = all(os.path.exists(p) for p in _FIXTURES)

requires_fixtures = pytest.mark.skipif(
    not _HAVE_FIXTURES,
    reason=(
        "requires grid_data/ and landsea_mask_data/ fixtures from "
        "script/generate_fractional_mask_via_ncremap_JCM_RGLL.sh (gitignored, not committed)"
    ),
)


# --- _make_esmf_regridder / flat-array adapter ------------------------------

@requires_fixtures
def test_regridder_indices_are_0_indexed_and_in_range():
    from EarthSystemGrids.base.esmf_regrid import ESMFRegridder

    r = ESMFRegridder(_JCM_TO_RGLL_CONSERVE)
    w = r.weights
    assert int(np.asarray(w.row_indices).min()) == 0
    assert int(np.asarray(w.row_indices).max()) == w.dst_size - 1
    assert int(np.asarray(w.col_indices).min()) == 0
    assert int(np.asarray(w.col_indices).max()) == w.src_size - 1
    assert w.src_size == 4608
    assert w.dst_size == 16200


@requires_fixtures
def test_regridder_constant_field_stays_constant_where_fully_covered():
    regridder = _make_esmf_regridder(_JCM_TO_RGLL_CONSERVE)
    ones = np.ones(4608, dtype=np.float32)
    out = np.asarray(regridder(ones))
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, 1.0, atol=1e-5)


@requires_fixtures
def test_regridder_matches_independent_scipy_reference():
    regridder = _make_esmf_regridder(_JCM_TO_RGLL_CONSERVE)

    with xr.open_dataset(_JCM_TO_RGLL_CONSERVE) as ds:
        row = ds["row"].values - 1
        col = ds["col"].values - 1
        S = ds["S"].values
        n_a = ds.sizes["n_a"]
        n_b = ds.sizes["n_b"]

    reference = scipy.sparse.coo_matrix((S, (row, col)), shape=(n_b, n_a)).tocsr()

    rng = np.random.default_rng(0)
    field = rng.random(n_a).astype(np.float64)

    expected = reference @ field
    actual = np.asarray(regridder(field))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


@requires_fixtures
def test_regridder_zero_coverage_cell_is_nan():
    # A destination index that never appears in `row` never receives any
    # weight, so frac_b for it is 0 -- confirm the adapter NaNs it out.
    with xr.open_dataset(_JCM_TO_RGLL_CONSERVE) as ds:
        row = ds["row"].values - 1
        frac_b = ds["frac_b"].values
    covered = np.zeros(frac_b.shape[0], dtype=bool)
    covered[row] = True
    # sanity: this weight file, from two global full-sphere grids, should
    # have every destination cell covered -- so cross-check against frac_b
    # directly rather than assuming an uncovered cell exists.
    assert np.array_equal(covered, frac_b > 0)

    regridder = _make_esmf_regridder(_JCM_TO_RGLL_CONSERVE)
    ones = np.ones(4608, dtype=np.float32)
    out = np.asarray(regridder(ones))
    zero_cov = np.flatnonzero(frac_b == 0)
    if zero_cov.size:
        assert np.isnan(out[zero_cov]).all()
    else:
        assert not np.isnan(out).any()


# --- check_coverage (module-level, no DomainMaster needed) -----------------

def test_check_coverage_detects_hole():
    valid = np.ones(10, dtype=bool)
    a = np.full(10, 1.0)
    a[:3] = np.nan
    b = np.full(10, 2.0)
    b[:3] = np.nan
    with pytest.raises(CoverageError):
        check_coverage(valid, {"a": a, "b": b})


def test_check_coverage_detects_overlap_by_default():
    valid = np.ones(10, dtype=bool)
    a = np.full(10, 1.0)
    b = np.full(10, 2.0)  # fully overlaps a everywhere
    with pytest.raises(CoverageError):
        check_coverage(valid, {"a": a, "b": b})
    report = check_coverage(valid, {"a": a, "b": b}, allow_overlap=True)
    assert report.ok


def test_check_coverage_passes_for_exact_partition():
    valid = np.ones(10, dtype=bool)
    a = np.full(10, 1.0)
    a[5:] = np.nan
    b = np.full(10, 2.0)
    b[:5] = np.nan
    report = check_coverage(valid, {"a": a, "b": b})
    assert report.ok
    assert not report.holes.any()
    assert not report.overlaps.any()


# --- DomainMaster ------------------------------------------------------------

@requires_fixtures
def test_register_domain_path_and_object_equivalent():
    from EarthSystemGrids.base.StructuredQuadMesh import StructuredQuadMesh

    dm = DomainMaster()
    d_from_path = dm.register_domain("JCM_path", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)

    mesh = StructuredQuadMesh.from_SCRIP_file(_JCM_SCRIP)
    with xr.open_dataset(_JCM_LANDSEA) as ds:
        mask_arr = ds["lsm"].squeeze().values.reshape(-1)
    d_from_object = dm.register_domain("JCM_object", mesh, landsea_mask=mask_arr)

    np.testing.assert_array_equal(d_from_path.landsea_mask, d_from_object.landsea_mask)
    np.testing.assert_allclose(d_from_path.grid.face_lon, d_from_object.grid.face_lon)


@requires_fixtures
def test_register_domain_topography_from_same_file_as_landsea_mask():
    dm = DomainMaster()
    d = dm.register_domain(
        "JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA, topography=_JCM_LANDSEA,
    )
    with xr.open_dataset(_JCM_LANDSEA) as ds:
        expected_mask = ds["lsm"].squeeze().values.reshape(-1)
        expected_topo = ds["topography"].squeeze().values.reshape(-1)
    np.testing.assert_array_equal(d.landsea_mask, expected_mask)
    np.testing.assert_array_equal(d.topography, expected_topo)


def _make_domain_master_with_domains():
    dm = DomainMaster()
    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_domain("RGLL", _RGLL_SCRIP, landsea_mask=_RGLL_LANDSEA, is_exchange_grid=True)
    return dm


@requires_fixtures
def test_exchange_grid_names():
    dm = _make_domain_master_with_domains()
    assert dm.exchange_grid_names == ["RGLL"]


@requires_fixtures
def test_register_transformation_lazy_loads_and_caches():
    dm = _make_domain_master_with_domains()
    t = dm.register_transformation(
        "JCM", "RGLL", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE
    )
    assert t.regridder is None

    dm.transform_scalar("JCM", "RGLL", "conserve", np.ones(4608, dtype=np.float32))
    built = t.regridder
    assert built is not None

    dm.transform_scalar("JCM", "RGLL", "conserve", np.ones(4608, dtype=np.float32))
    assert t.regridder is built  # cached, not rebuilt


@requires_fixtures
def test_register_transformation_does_not_require_registered_domains():
    # Registration is lazy: source/target don't need to exist yet -- see
    # test_register_transformation_before_domains_exist_does_not_raise and
    # the validate() tests below for where a still-dangling reference is
    # actually caught.
    dm = DomainMaster()
    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_transformation("JCM", "nope", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE)
    dm.register_transformation("nope", "JCM", "conserve2", weight_file=_JCM_TO_RGLL_CONSERVE)


@requires_fixtures
def test_register_transformation_rejects_both_weight_file_and_regridder():
    dm = DomainMaster()
    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_domain("RGLL", _RGLL_SCRIP, landsea_mask=_RGLL_LANDSEA)
    with pytest.raises(ValueError):
        dm.register_transformation(
            "JCM", "RGLL", "conserve",
            weight_file=_JCM_TO_RGLL_CONSERVE, regridder=lambda f: f,
        )


@requires_fixtures
def test_register_transformation_placeholder_lifecycle():
    # Neither weight_file nor regridder registers a placeholder -- the
    # "declare the transformation graph now, fill in the mechanism later"
    # workflow.
    dm = DomainMaster()
    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_domain("RGLL", _RGLL_SCRIP, landsea_mask=_RGLL_LANDSEA)
    dm.register_transformation("JCM", "RGLL", "conserve")

    t = next(iter(dm.transformations.values()))
    assert t.weight_file is None and t.regridder is None
    with pytest.raises(ValueError):
        dm.transform_scalar("JCM", "RGLL", "conserve", np.ones(4608))
    problems = dm.validate(raise_error=False)
    assert len(problems) == 1 and "placeholder" in problems[0]

    # filling in a placeholder doesn't need overwrite=True
    dm.register_transformation(
        "JCM", "RGLL", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE
    )
    assert dm.validate(raise_error=False) == []

    # but re-registering an already-resolved transformation still does
    with pytest.raises(ValueError):
        dm.register_transformation("JCM", "RGLL", "conserve", regridder=lambda f: f)
    dm.register_transformation(
        "JCM", "RGLL", "conserve", regridder=lambda f: f, overwrite=True
    )
    t = next(iter(dm.transformations.values()))
    assert t.weight_file is None and t.regridder is not None


@requires_fixtures
def test_transform_scalar_unregistered_transformation_raises():
    dm = _make_domain_master_with_domains()
    with pytest.raises(KeyError):
        dm.transform_scalar("JCM", "RGLL", "nope", np.zeros(4608))


@requires_fixtures
def test_transform_scalar_custom_regridder_used_directly():
    dm = _make_domain_master_with_domains()
    dm.register_transformation("JCM", "RGLL", "double", regridder=lambda f: 2.0 * np.asarray(f)[:5])
    out = dm.transform_scalar("JCM", "RGLL", "double", np.arange(4608, dtype=np.float64))
    np.testing.assert_array_equal(out, np.array([0.0, 2.0, 4.0, 6.0, 8.0]))


@requires_fixtures
def test_transform_scalar_conserve_end_to_end_constant_field():
    dm = _make_domain_master_with_domains()
    dm.register_transformation("JCM", "RGLL", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE)
    ones = np.ones(4608, dtype=np.float32)
    out = np.asarray(dm.transform_scalar("JCM", "RGLL", "conserve", ones))
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, 1.0, atol=1e-5)


@requires_fixtures
def test_transform_scalar_conserve_is_area_conservative():
    dm = _make_domain_master_with_domains()
    dm.register_transformation("JCM", "RGLL", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE)

    with xr.open_dataset(_JCM_SCRIP) as ds:
        src_area = ds["grid_area"].values
    with xr.open_dataset(_RGLL_SCRIP) as ds:
        dst_area = ds["grid_area"].values

    rng = np.random.default_rng(0)
    field = rng.random(4608).astype(np.float32)
    out = np.asarray(dm.transform_scalar("JCM", "RGLL", "conserve", field))

    src_integral = float(np.sum(field * src_area))
    dst_integral = float(np.sum(out * dst_area))
    rel_err = abs(dst_integral - src_integral) / abs(src_integral)
    assert rel_err < 1e-4


@requires_fixtures
def test_transform_vector_matches_manual_rotation():
    dm = _make_domain_master_with_domains()
    dm.register_transformation("JCM", "RGLL", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE)

    rng = np.random.default_rng(1)
    u = rng.random(4608).astype(np.float32)
    v = rng.random(4608).astype(np.float32)

    u_dst, v_dst = dm.transform_vector("JCM", "RGLL", "conserve", u, v)

    cos_src, sin_src = _angle_trig(dm._domains["JCM"].grid)
    cos_dst, sin_dst = _angle_trig(dm._domains["RGLL"].grid)
    u_east = u * cos_src - v * sin_src
    v_north = u * sin_src + v * cos_src
    u_east_dst = np.asarray(dm.transform_scalar("JCM", "RGLL", "conserve", u_east))
    v_north_dst = np.asarray(dm.transform_scalar("JCM", "RGLL", "conserve", v_north))
    expected_u = u_east_dst * cos_dst + v_north_dst * sin_dst
    expected_v = -u_east_dst * sin_dst + v_north_dst * cos_dst

    np.testing.assert_allclose(np.asarray(u_dst), expected_u, atol=1e-5)
    np.testing.assert_allclose(np.asarray(v_dst), expected_v, atol=1e-5)


@requires_fixtures
def test_transform_vector_reduces_to_identity_for_identity_regridder_same_grid():
    dm = DomainMaster()
    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_transformation("JCM", "JCM", "identity", regridder=lambda f: f)

    rng = np.random.default_rng(2)
    u = rng.random(4608)
    v = rng.random(4608)
    u_out, v_out = dm.transform_vector("JCM", "JCM", "identity", u, v)
    # source == target -> the source->geographic and geographic->target
    # rotations exactly cancel, so this must reduce to the identity.
    np.testing.assert_allclose(np.asarray(u_out), u, atol=1e-10)
    np.testing.assert_allclose(np.asarray(v_out), v, atol=1e-10)


@requires_fixtures
def test_transform_vector_raises_for_non_structured_grid():
    dm = _make_domain_master_with_domains()

    jcm_mesh = dm._domains["JCM"].grid
    lon = jcm_mesh.node_lon[:4]
    lat = jcm_mesh.node_lat[:4]
    face_corner_lon = np.array([[lon[0], lon[1], lon[2]], [lon[0], lon[2], lon[3]]])
    face_corner_lat = np.array([[lat[0], lat[1], lat[2]], [lat[0], lat[2], lat[3]]])
    face_lon = face_corner_lon.mean(axis=1)
    face_lat = face_corner_lat.mean(axis=1)
    poly_mesh = UnstructuredGridMesh.from_polygons(
        face_corner_lon, face_corner_lat, face_lon, face_lat,
        area=np.array([1.0, 1.0]), mask=np.array([1, 1]),
    )
    dm.register_domain("POLY", poly_mesh, landsea_mask=np.array([0.5, 0.5]))
    dm.register_transformation("JCM", "POLY", "nearest", regridder=lambda f: f[:2])

    with pytest.raises(TypeError):
        dm.transform_vector("JCM", "POLY", "nearest", np.zeros(4608), np.zeros(4608))


@requires_fixtures
def test_domain_master_check_coverage_detects_hole_and_overlap():
    dm = _make_domain_master_with_domains()
    n_b = dm._domains["RGLL"].grid.face_lon.size
    valid = np.ones(n_b, dtype=bool)

    a_with_hole = np.full(n_b, 1.0)
    a_with_hole[:10] = np.nan
    b_with_hole = np.full(n_b, 2.0)
    b_with_hole[:10] = np.nan
    with pytest.raises(CoverageError):
        dm.check_coverage("RGLL", {"a": a_with_hole, "b": b_with_hole}, valid_mask=valid)

    a_full = np.full(n_b, 1.0)
    b_full = np.full(n_b, 2.0)
    with pytest.raises(CoverageError):
        dm.check_coverage("RGLL", {"a": a_full, "b": b_full}, valid_mask=valid)
    report = dm.check_coverage(
        "RGLL", {"a": a_full, "b": b_full}, valid_mask=valid, allow_overlap=True
    )
    assert report.ok


@requires_fixtures
def test_domain_master_check_coverage_default_valid_mask_from_grid_mask():
    dm = _make_domain_master_with_domains()
    n_b = dm._domains["RGLL"].grid.face_lon.size
    contribution = np.full(n_b, 1.0)
    report = dm.check_coverage("RGLL", {"only_source": contribution})
    np.testing.assert_array_equal(report.valid_mask, dm._domains["RGLL"].grid.mask == 1)


# --- lazy registration, inspection, and validation --------------------------

@requires_fixtures
def test_register_transformation_before_domains_exist_does_not_raise():
    dm = DomainMaster()
    # neither "JCM" nor "RGLL" is registered yet
    t = dm.register_transformation(
        "JCM", "RGLL", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE
    )
    assert t.source == "JCM" and t.target == "RGLL"

    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_domain("RGLL", _RGLL_SCRIP, landsea_mask=_RGLL_LANDSEA)

    # now resolvable
    out = dm.transform_scalar("JCM", "RGLL", "conserve", np.ones(4608, dtype=np.float32))
    assert np.asarray(out).shape == (16200,)
    assert dm.validate(raise_error=False) == []


def test_repr_lists_domains_and_flags_missing_transformation():
    dm = DomainMaster()
    if not _HAVE_FIXTURES:
        pytest.skip("requires fixtures")
    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_transformation("JCM", "RGLL", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE)

    text = repr(dm)
    assert "'JCM'" in text
    assert "MISSING" in text
    assert "target 'RGLL' not registered" in text
    # print(dm) exercises __str__, which should delegate to __repr__
    assert str(dm) == text


@requires_fixtures
def test_domains_and_transformations_properties_are_read_only_views():
    dm = DomainMaster()
    dm.register_transformation("JCM", "RGLL", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE)

    assert dm.domains == {}
    assert list(dm.transformations.keys()) == [("JCM", "RGLL", "conserve")]
    t = dm.transformations[("JCM", "RGLL", "conserve")]
    assert t.source == "JCM" and t.target == "RGLL" and t.method == "conserve"
    assert t.weight_file == _JCM_TO_RGLL_CONSERVE
    assert t.regridder is None  # not built yet
    with pytest.raises(TypeError):
        dm.domains["hack"] = None
    with pytest.raises(TypeError):
        dm.transformations[("a", "b", "c")] = None

    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_domain("RGLL", _RGLL_SCRIP)  # landsea_mask left to default

    assert dm.domains["JCM"].attrs["landsea_mask_provided"] is True
    assert dm.domains["RGLL"].attrs["landsea_mask_provided"] is False
    assert type(dm.domains["JCM"].grid).__name__ == "StructuredQuadMesh"
    assert dm.domains["JCM"].grid.face_lon.size == 4608

    assert dm.transformations[("JCM", "RGLL", "conserve")].regridder is None

    dm.transform_scalar("JCM", "RGLL", "conserve", np.ones(4608, dtype=np.float32))
    assert dm.transformations[("JCM", "RGLL", "conserve")].regridder is not None


@requires_fixtures
def test_validate_raises_for_dangling_reference_then_passes_once_resolved():
    dm = DomainMaster()
    dm.register_transformation("JCM", "RGLL", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE)

    with pytest.raises(ValidationError):
        dm.validate()
    problems = dm.validate(raise_error=False)
    assert len(problems) == 1
    assert "JCM" in problems[0] and "not registered" in problems[0]

    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_domain("RGLL", _RGLL_SCRIP, landsea_mask=_RGLL_LANDSEA)
    assert dm.validate(raise_error=False) == []
    dm.validate()  # does not raise


@requires_fixtures
def test_validate_catches_weight_file_domain_size_mismatch():
    dm = DomainMaster()
    # Both "A" and "B" are JCM-sized (4608), but the registered weight file
    # is JCM (4608) -> RGLL (16200) -- a genuine size mismatch on the target.
    dm.register_domain("A", _JCM_SCRIP)
    dm.register_domain("B", _JCM_SCRIP)
    dm.register_transformation("A", "B", "conserve", weight_file=_JCM_TO_RGLL_CONSERVE)

    problems = dm.validate(raise_error=False)
    assert len(problems) == 1
    assert "does not match target domain" in problems[0]
    with pytest.raises(ValidationError):
        dm.validate()


@requires_fixtures
def test_validate_catches_custom_regridder_wrong_output_size():
    dm = _make_domain_master_with_domains()
    dm.register_transformation("JCM", "RGLL", "bad", regridder=lambda f: np.asarray(f)[:5])

    problems = dm.validate(raise_error=False)
    assert len(problems) == 1
    assert "does not match target domain" in problems[0]


# --- lazy grid registration --------------------------------------------------

def test_register_domain_name_only_is_a_placeholder():
    dm = DomainMaster()
    d = dm.register_domain("atm")
    assert d.grid is None
    assert d.landsea_mask is None
    assert d.topography is None
    assert dm.domains["atm"] is d


def test_register_domain_rejects_landsea_mask_without_grid():
    dm = DomainMaster()
    with pytest.raises(ValueError):
        dm.register_domain("atm", landsea_mask=np.array([0.5]))


@requires_fixtures
def test_register_domain_placeholder_filled_in_without_overwrite():
    dm = DomainMaster()
    dm.register_domain("JCM", is_exchange_grid=True)
    assert dm.domains["JCM"].grid is None

    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    assert dm.domains["JCM"].grid is not None
    assert dm.domains["JCM"].grid.face_lon.size == 4608
    # filling in a placeholder doesn't carry over the placeholder call's
    # is_exchange_grid -- each register_domain call fully specifies state
    assert dm.domains["JCM"].is_exchange_grid is False

    # but re-registering the now-resolved domain still needs overwrite=True
    with pytest.raises(ValueError):
        dm.register_domain("JCM", _JCM_SCRIP)
    dm.register_domain("JCM", _JCM_SCRIP, overwrite=True)


def test_repr_flags_placeholder_domain_and_dependent_transformation():
    dm = DomainMaster()
    dm.register_domain("atm")
    dm.register_domain("ocn")
    dm.register_transformation("atm", "ocn", "bilinear")

    text = repr(dm)
    assert "domain 'atm': UNRESOLVED (no grid set yet)" in text
    assert "source 'atm' has no grid set yet" in text
    assert "target 'ocn' has no grid set yet" in text


def test_validate_catches_placeholder_domain_referenced_by_transformation():
    dm = DomainMaster()
    dm.register_domain("atm")
    dm.register_domain("ocn")
    dm.register_transformation("atm", "ocn", "bilinear", regridder=lambda f: f)

    problems = dm.validate(raise_error=False)
    assert len(problems) == 1
    assert "atm" in problems[0] and "no grid set yet" in problems[0]


@requires_fixtures
def test_transform_vector_raises_clearly_for_domain_with_no_grid():
    dm = DomainMaster()
    dm.register_domain("JCM", _JCM_SCRIP, landsea_mask=_JCM_LANDSEA)
    dm.register_domain("ocn")
    dm.register_transformation("JCM", "ocn", "bilinear", regridder=lambda f: f)

    with pytest.raises(TypeError, match="no grid set yet"):
        dm.transform_vector("JCM", "ocn", "bilinear", np.zeros(4608), np.zeros(4608))


@requires_fixtures
def test_check_coverage_raises_clearly_when_exchange_domain_has_no_grid():
    dm = DomainMaster()
    dm.register_domain("ocn")
    with pytest.raises(ValueError, match="no grid set yet"):
        dm.check_coverage("ocn", {"a": np.zeros(10)})
