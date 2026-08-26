from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.collections import PolyCollection

from EarthSystemGrids.base import StructuredQuadMesh
from EarthSystemGrids.domain_master import DomainMaster, make_mask_regridder
from EarthSystemGrids.RotatedGaussianLatLon import RotatedGaussianLatLon
from EarthSystemGrids import GaussianLatLon

import xarray as xr

grid_dir = Path("grid_data")


atm_grid = StructuredQuadMesh.from_SCRIP_file(grid_dir / "JCM_T31.SCRIP.nc")
sfc_grid = StructuredQuadMesh.from_SCRIP_file(grid_dir / "RotatedGaussianLatLon.SCRIP.nc")

sfc_lat = np.asarray(sfc_grid.face_lat)
sfc_lon = np.asarray(sfc_grid.face_lon)
sfc_land_fraction = jnp.asarray(xr.open_dataset(grid_dir / "landsea_mask_fraction_RotatedGaussianLatLon.nc")["lsm"].isel(valid_time=0).to_numpy())
mask_threshold = 0.3


dm = DomainMaster()
dm.register_domain(
    name = "atm",
    grid = atm_grid,
    mask = jnp.ones(atm_grid.shape),
    land_fraction = jnp.zeros(atm_grid.shape),
    topography = jnp.zeros(atm_grid.shape),
)

dm.register_domain(
    name = "exchange",
    grid = sfc_grid,
    is_exchange_grid = True,
    mask = jnp.zeros(sfc_grid.shape),
)


dm.register_domain(
    name = "ocn",
    grid = sfc_grid,
    land_fraction = sfc_land_fraction,
    mask = sfc_land_fraction < mask_threshold,
)

dm.register_domain(
    name = "lnd",
    grid = sfc_grid,
    land_fraction = sfc_land_fraction,
    mask = sfc_land_fraction >= mask_threshold,
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


# =============================================================================
# Exercise the atm -> exchange -> ocn/lnd regridding chain with a synthetic
# field, and verify the ocn/lnd masks partition the exchange grid exactly.
# =============================================================================

field_atm = np.sin(3.0 * np.asarray(atm_grid.face_lon)) * np.cos(2.0 * np.asarray(atm_grid.face_lat))

field_exchange = np.asarray(dm.transform_scalar("atm", "exchange", "conserve", field_atm))
field_ocn = np.asarray(dm.transform_scalar("exchange", "ocn", "identity", field_exchange))
field_lnd = np.asarray(dm.transform_scalar("exchange", "lnd", "identity", field_exchange))

# Verify: does every exchange-grid cell get covered by exactly one of
# ocn/lnd, with no holes and no overlaps? Regrid an all-ones field through
# each mask transformation and hand both to check_coverage -- this is the
# same NaN-means-uncovered convention transform_scalar itself already uses.
ones_exchange = np.ones(sfc_grid.shape).reshape(-1)
coverage = dm.check_coverage(
    "exchange",
    {
        "ocn": dm.transform_scalar("exchange", "ocn", "identity", ones_exchange),
        "lnd": dm.transform_scalar("exchange", "lnd", "identity", ones_exchange),
    },
    raise_error=False,
)
print(
    f"mask coverage check: ok={coverage.ok}, "
    f"holes={int(coverage.holes.sum())}, overlaps={int(coverage.overlaps.sum())} "
    f"(out of {int(coverage.valid_mask.sum())} valid exchange cells)"
)

# 0 = fine, 1 = hole (covered by neither ocn nor lnd), 2 = overlap (covered
# by both) -- plotted below so a mask bug would show up as a visible smear
# rather than just a count.
coverage_status = np.zeros(sfc_grid.shape).reshape(-1)
coverage_status[coverage.holes] = 1
coverage_status[coverage.overlaps] = 2


# =============================================================================
# Plotting
# =============================================================================

def _face_polygons(mesh):
    """
    (nface, 4, 2) lon/lat corner vertices in degrees, for a StructuredQuadMesh.

    Each face's corners are unwrapped to the branch nearest that face's own
    centre longitude (`lon - 360*round((lon-center)/360)`) so cells that
    straddle the dateline/prime-meridian seam don't get drawn as a
    smear spanning the whole plot.
    """
    corner_lon = np.rad2deg(np.asarray(mesh.node_lon)[mesh.face_nodes])
    corner_lat = np.rad2deg(np.asarray(mesh.node_lat)[mesh.face_nodes])
    center_lon = np.rad2deg(np.asarray(mesh.face_lon))[:, None]
    corner_lon = corner_lon - 360.0 * np.round((corner_lon - center_lon) / 360.0)
    return np.stack([corner_lon, corner_lat], axis=-1)


def plot_field(ax, mesh, field, *, title=None, cmap="RdBu_r", vmin=None, vmax=None):
    """Plot a (nface,) field as filled grid cells on a cartopy PlateCarree axis."""
    verts = _face_polygons(mesh)
    pc = PolyCollection(
        verts, array=np.asarray(field).reshape(-1),
        cmap=cmap, transform=ccrs.PlateCarree(), edgecolors="none",
    )
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_global()
    ax.coastlines(linewidth=0.5, color="black")
    if title:
        ax.set_title(title, fontsize=9)
    return pc


vmin, vmax = float(field_atm.min()), float(field_atm.max())
proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(14, 8))

ax_atm      = fig.add_subplot(2, 3, 1, projection=proj)
ax_exchange = fig.add_subplot(2, 3, 2, projection=proj)
ax_coverage = fig.add_subplot(2, 3, 3, projection=proj)
ax_ocn      = fig.add_subplot(2, 3, 5, projection=proj)
ax_lnd      = fig.add_subplot(2, 3, 6, projection=proj)

pc_atm = plot_field(ax_atm, atm_grid, field_atm, title="atm (source field)", vmin=vmin, vmax=vmax)
fig.colorbar(pc_atm, ax=ax_atm, orientation="horizontal", pad=0.05, shrink=0.8)

plot_field(ax_exchange, sfc_grid, field_exchange, title="exchange (atm→exchange, conserve)", vmin=vmin, vmax=vmax)
plot_field(ax_ocn, sfc_grid, field_ocn, title="ocn (exchange→ocn, masked)", vmin=vmin, vmax=vmax)
plot_field(ax_lnd, sfc_grid, field_lnd, title="lnd (exchange→lnd, masked)", vmin=vmin, vmax=vmax)

plot_field(
    ax_coverage, sfc_grid, coverage_status,
    title=f"ocn/lnd mask coverage (ok={coverage.ok})\n0=fine 1=hole 2=overlap",
    cmap="RdYlGn_r", vmin=0, vmax=2,
)

fig.suptitle("atm → exchange → ocn/lnd regridding")
fig.tight_layout()

out_path = Path("use_domain_master.png")
fig.savefig(out_path, dpi=150)
print(f"wrote {out_path}")

plt.show()
