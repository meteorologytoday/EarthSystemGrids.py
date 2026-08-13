import numpy as np

from EarthSystemGrids.base import StructuredQuadMesh, _R_EARTH
from EarthSystemGrids import GaussianLatLon


def _horizontal_grid_for_resolution(resolution: int):
    """
    The dinosaur.spherical_harmonic.Grid for a JCM spectral truncation.

    Mirrors jcm.utils.get_coords's own horizontal-grid selection, without
    the vertical-coordinate / physics machinery that function also builds:
    JCMGrid only needs the horizontal grid's node coordinates, not a full
    model CoordinateSystem. T63 has no dedicated
    dinosaur.spherical_harmonic.Grid.T63 factory, so it is built directly
    via Grid.construct with the same (max_wavenumber=63, gaussian_nodes=48)
    pairing jcm.utils.get_coords uses, which reproduces the standard
    192 x 96 ECHAM T63 grid.
    """
    import dinosaur.spherical_harmonic as dsh
    from jcm.utils import VALID_TRUNCATIONS

    if resolution not in VALID_TRUNCATIONS:
        raise ValueError(
            f"Invalid resolution (spectral truncation) {resolution}. "
            f"Must be one of: {VALID_TRUNCATIONS}."
        )
    if resolution == 63:
        return dsh.Grid.construct(max_wavenumber=63, gaussian_nodes=48)
    return getattr(dsh.Grid, f"T{resolution}")()


def generate_JCMGrid(resolution: int, mask=None, earth_radius: float = _R_EARTH,
                     attrs=None) -> StructuredQuadMesh:
    """
    Build the JCM spectral-truncation grid as a StructuredQuadMesh.

    Delegates the actual mesh construction to GaussianLatLon.generate_mesh,
    passing jcm's own native grid centres (Gaussian latitudes, uniformly
    spaced longitudes) as explicit degree arrays rather than regenerating
    them independently -- this keeps the mesh tied to the exact values the
    model itself uses (see _horizontal_grid_for_resolution), not values
    that merely ought to equal them.

    Parameters
    ----------
    resolution   : spectral truncation, e.g. 31 for T31; see jcm.utils.VALID_TRUNCATIONS
    mask         : (nlat, nlon) int, optional. 0=land 1=ocean. Defaults to all ocean.
    earth_radius : metres, used only to convert solid angle into m² area.
    attrs        : dict of grid-specific metadata.

    Returns
    -------
    StructuredQuadMesh, shape (nlat, nlon)
    """
    horizontal_grid = _horizontal_grid_for_resolution(resolution)
    lat_centers_deg = np.rad2deg(np.asarray(horizontal_grid.latitudes))
    lon_centers_deg = np.rad2deg(np.asarray(horizontal_grid.longitudes))

    grid_attrs = {"title": f"JCM T{resolution} spectral-truncation grid"}
    grid_attrs.update(attrs or {})

    return GaussianLatLon.generate_mesh(
        lat          = lat_centers_deg,
        lon          = lon_centers_deg,
        mask         = mask,
        earth_radius = earth_radius,
        attrs        = grid_attrs,
    )


def test_output_SCRIP_file():
    output_file = "grid_JCM.nc"

    print("Generating grid...")
    mesh = generate_JCMGrid(31)

    print("Writing to file: ", output_file)
    mesh.write_to_SCRIP_grid_file(output_file, flatten=False)

if __name__ == "__main__":
    test_output_SCRIP_file()
