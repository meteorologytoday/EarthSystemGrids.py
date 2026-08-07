import numpy as np
import scipy.integrate as itg

"""
Displaced Pole Grid

Reference:

  -

"""

def function_f(s_j:float):
    """
    s_j in [-pi/2, pi/2] is the mesh latitude.
    f has to be decreasing when s_j increases.
    """

    return np.tan(np.pi/4 - s_j/2)  #2.0 * ( -1.0 + s_j )

def function_dfds_j(s_j:float):
    """
    the derivative of f over s_j
    """
    return (np.cos(np.pi/4 - s_j/2))**(-2.0) * (-1/2)


def function_g(s_j:float):
    """
    s_j in [0, 1]. 0 is at mesh south pole, 0.5 is at the equator, and 1 is at
    the mesh north pole. g has to be increasing when s increases.
    """
    if s_j <= 0:
        return - function_f(s_j)
    else:
        return - np.tan(np.pi/4 - s_j/2)

def function_dgds_j(s_j:float):
    """
    the derivative of g over s_j
    """
    if s_j <= 0:
        return - function_dfds_j(s_j)
    else:
        return - function_dfds_j(s_j)

def I_curve_system(s, variables):

    x, y = variables

    f = function_f(s)
    g = function_g(s)
    df_ds = function_dfds_j(s)
    dg_ds = function_dgds_j(s)

    C = f + g
    P = f * g
    grad_phi_square = 4 * x**2 + (2 * y - C)**2
    dC_ds = df_ds + dg_ds
    dP_ds = df_ds * g + f * dg_ds

    dphi_ds = dP_ds - dC_ds * y

    dx_ds = - 2*x * (dphi_ds / grad_phi_square)
    dy_ds = - (2*y - C) * (dphi_ds / grad_phi_square)

    return np.array([
        dx_ds,
        dy_ds,
    ])

def project_stereo_to_sphere(x:float, y:float):
    """
    Project stereographic coordinate back to sphere coordinate.
    Unit circle means equator, and center means northpole.
    """
    lon = np.arctan2(y, x)
    lat = np.pi / 2.0 - 2 * np.arctan((x**2 + y**2)**0.5)

    return lon, lat

def generate_J_curve_from_s(
    s_j: float,
    number_of_points: int,
):
    """
    Compute the J curve (mesh zonal circle) given the mesh latitude s_j.
    """

    f = function_f(s_j)
    g = function_g(s_j)

    r   = (g - f) / 2.0
    y_c = (g + f) / 2.0

    dh = np.deg2rad(360.0 / number_of_points)
    h = 0.0 + np.arange(number_of_points) * dh

    pts = np.zeros((2, number_of_points))

    x =       r * np.cos(h)
    y = y_c + r * np.sin(h)

    lon, lat = project_stereo_to_sphere(x, y)
    pts[0, :] = lon
    pts[1, :] = lat

    return pts

def generate_I_curve_from_s(
    s_i: float,
    number_of_points: int,
):
    """
    Compute the I curve (mesh zonal circle) given the mesh longitude s_i.
    """

    # initial point: on the equator
    x0 = np.cos(s_i)
    y0 = np.sin(s_i)
    ds = np.pi/2 / number_of_points
    _, pts_xy = integrate_euler_forward(I_curve_system, 0.0, [x0, y0], ds, number_of_points-1)

    lon, lat = project_stereo_to_sphere(pts_xy[0, :], pts_xy[1, :])
    pts = np.zeros((2, number_of_points))
    pts[0, :] = lon
    pts[1, :] = lat

    return pts


def generate_I_curve(
    s_i: float,
    J_eq: int,
    J_M: int,
    split: int, 
):
    """
    Compute the I curve (mesh zonal circle) given the mesh longitude s_i.
    """
    # initial point: on the equator
    x0 = np.cos(s_i)
    y0 = np.sin(s_i)
    number_of_grids = J_M - J_eq
    integration_steps = number_of_grids * split 
    ds = (np.pi/2) / integration_steps
    _, pts_xy = integrate_euler_forward(I_curve_system, 0.0, [x0, y0], ds, integration_steps)

    lon, lat = project_stereo_to_sphere(pts_xy[0, ::split], pts_xy[1, ::split])
    pts = np.zeros((2, number_of_grids + 1))
    pts[0, :] = lon
    pts[1, :] = lat

    return pts

def spherical_to_cartesian(
    r_sphere,
    r:float = 1.0,
    lon_idx:int = 0,
    lat_idx:int = 1,
):
    """
    Convert from spherical coordinate to lonlat
    The 0th axis is (x, y, z), the 1st axis is the points
    """
    lon = r_sphere[lon_idx, :]
    lat = r_sphere[lat_idx, :]
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.stack((x, y, z), axis=0)

def integrate_euler_forward(dydt, t0, y0, dt, steps):

    dim = len(y0)
    y = np.zeros((dim, steps+1))
    t = np.zeros((steps+1,))
    t[0] = t0
    y[:, 0] = y0
    for step in range(steps):

        t[step+1] = t[step] + dt
        y[:, step+1] = y[:, step] + dydt(t[step], y[:, step]) * dt

    return t, y



if __name__ == "__main__":

    J_curves = [spherical_to_cartesian(generate_J_curve_from_s(s, 50)) for s in np.linspace(-np.pi/2, np.pi/2, 10)]
    I_curves = [spherical_to_cartesian(generate_I_curve(s_i, 20, 60, 1)) for s_i in np.linspace(0.0, 2*np.pi, 10)[::-1]]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.view_init(azim=-30, elev=45, roll=0)

    ax.scatter(0, 0, 0, color="red", s=10)

    for i, I_curve in enumerate(I_curves):
        ax.scatter(I_curve[0, :], I_curve[1, :], I_curve[2, :])

    for j, J_curve in enumerate(J_curves):
        ax.scatter(J_curve[0, :], J_curve[1, :], J_curve[2, :], marker="s")


    ax.set_xlabel("x-direction")
    ax.set_ylabel("y-direction")
    ax.set_zlabel("z-direction")

    lim = np.array([-1, 1])*1.5
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)
    plt.show()


