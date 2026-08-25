from pathlib import Path

from EarthSystemGrids.base import StructuredQuadMesh
from EarthSystemGrids.domain_master import DomainMaster, make_mask_regridder
from EarthSystemGrids.RotatedGaussianLatLon import RotatedGaussianLatLon
from EarthSystemGrids import GaussianLatLon

"""
atm_grid = GaussianLatLon.generate_mesh(
    lat_bounds = GaussianLatLon.equally_spaced_bounds(18, -90.0,  90.0),
    lon_bounds = GaussianLatLon.equally_spaced_bounds(36,   0.0, 360.0),
)

sfc_grid = RotatedGaussianLatLon.generate_mesh(
    lat=GaussianLatLon.equally_spaced_bounds(120, -90.0, 90.0),
    lon=GaussianLatLon.equally_spaced_bounds(240, 0.0, 360.0),
    rotation_axis_longitude_deg=48.0,
    rotation_deg=12.0,
)
"""
grid_dir = Path("grid_data")


atm_grid = StructuredQuadMesh.from_SCRIP_file(grid_dir / "JCM_T31.SCRIP.nc")
sfc_grid = StructuredQuadMesh.from_SCRIP_file(grid_dir / "RotatedGaussianLatLon.SCRIP.nc")

dm = DomainMaster()
dm.register_domain(
    name = "atm",
    grid = atm_grid,
#    landsea_mask = ,
#    topography = ,
)

dm.register_domain(
    name = "exchange",
    grid = sfc_grid,
    is_exchange_grid = True,
)


dm.register_domain(
    name = "ocn",
    grid = sfc_grid,
#    landsea_mask = ,
#    topography = ,
)

dm.register_domain(
    name = "lnd",
    grid = sfc_grid,
#    landsea_mask = ,
#    topography = ,
)

dm.register_transformation(source="atm", target="exchange", method="bilinear", weight_file=grid_dir / "weight_algo-bilinear_JCM_T31_to_RotatedGaussianLatLon.nc")
dm.register_transformation(source="atm", target="exchange", method="conserve", weight_file=grid_dir / "weight_algo-conserve_JCM_T31_to_RotatedGaussianLatLon.nc")

dm.register_transformation(source="exchange", target="atm", method="bilinear", weight_file=grid_dir / "weight_algo-bilinear_RotatedGaussianLatLon_to_JCM_T31.nc")
dm.register_transformation(source="exchange", target="atm", method="conserve", weight_file=grid_dir / "weight_algo-conserve_RotatedGaussianLatLon_to_JCM_T31.nc")

dm.register_transformation(source="exchange", target="ocn", method="identity", regridder=make_mask_regridder(lambda: dm.domains["ocn"].mask))
dm.register_transformation(source="exchange", target="lnd", method="identity", regridder=make_mask_regridder(lambda: dm.domains["lnd"].mask))



print(dm)

problems = dm.validate(raise_error=False)
if len(problems) == 0:
    print("There is no problem detected!")
else:
    for i, problem in enumerate(problems):
        print(f"Problem {i+1:d} : {str(problem):s}")

