import numpy as np
import functools

def spherical_to_stereo(lon:float, lat:float):
    """
    Project sphere coordinate to stereographic.
    Unit circle means equator, and center means northpole.
    """
    rho = np.tan(np.pi/4 - lat/2)
    x = rho * np.cos(lon)
    y = rho * np.sin(lon)
    return x, y


def project_stereo_to_sphere(x:float, y:float):
    """
    Project stereographic coordinate back to sphere coordinate.
    Unit circle means equator, and center means northpole.
    """
    lon = np.arctan2(y, x)
    lat = np.pi / 2.0 - 2 * np.arctan((x**2 + y**2)**0.5)

    return lon, lat

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

def integrate_rk4(dydt, t0, y0, t_targets, substeps=25):
    """
    March to each entry of `t_targets` in turn, sub-stepping in between.

    The step size is decoupled from the row spacing, so the rows can be placed
    wherever the design wants them, and the integrator never has to take a step
    comparable to the shrinking radius near the pole.
    """
    y = np.array(y0, dtype=float)
    t = float(t0)
    out = np.zeros((len(y0), len(t_targets)))
    for k, t_end in enumerate(t_targets):
        dt = (t_end - t)/substeps
        for _ in range(substeps):
            k1 = dydt(t,          y)
            k2 = dydt(t + dt/2.0, y + dt/2.0*k1)
            k3 = dydt(t + dt/2.0, y + dt/2.0*k2)
            k4 = dydt(t + dt,     y + dt*k3)
            y = y + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
            t = t + dt
        t = t_end                        # remove accumulated drift
        out[:, k] = y
    return t, out


def I_curve_ray_tracing_system(
    v,
    variables,
    FG_funcs,
):
    """
    Right-hand side of the ODE that traces a single I-curve (mesh meridian).

    The mesh parallels (J-curves) are the zero level sets of the potential

        (1) phi(x, y, v) = x^2 + y^2 - ( f(v) + g(v) ) y + f(v) g(v)

    where f and g are functions of the pseudo-latitude v. The J-curve labelled
    by v is therefore

        (2) phi(x, y, v) = 0

    For a fixed v, (2) is a level set of phi, so the gradient

            grad phi = ( 2x , 2y - (f+g) )

    is normal to that J-curve. An I-curve must cross every J-curve at a right
    angle, so its tangent is parallel to grad phi,

        (3) (dx/dv, dy/dv) = m * grad phi

    for some scalar m = m(v) yet to be determined.

    # Finding m

    A point riding the moving contour stays on it for every v, so the TOTAL
    derivative of (2) along the trajectory vanishes,

        (4) grad phi dot (dx/dv, dy/dv)  +  d_v phi = 0

    where d_v phi is the PARTIAL derivative taken at frozen (x, y). Substituting
    (3) into (4) gives m |grad phi|^2 + d_v phi = 0, hence

        (5) m = - d_v phi / |grad phi|^2

    # The ray-tracing system

        (6) (dx/dv, dy/dv) = - d_v phi / |grad phi|^2 * ( 2x , 2y - (f+g) )

            |grad phi|^2   = 4 x^2 + ( 2y - (f+g) )^2

    Integrating (6) in v, from a starting point on the equator circle, produces
    the complete I-curve.

    # Notes

    Degeneracy. |grad phi|^2 vanishes only at x = 0, y = (f+g)/2, the centre of
    the circle. That point lies ON the circle only when the radius is zero, i.e.
    at the mesh north pole. Boundary conditions that close the pole exactly,
    f(90deg) = g(90deg), therefore make (6) a 0/0 on the last row, and the
    integration has to stop short of v_max.
    """
    x, y = variables

    f = FG_funcs.f(v)
    g = FG_funcs.g(v)
    df_dv = FG_funcs.dfdv(v)
    dg_dv = FG_funcs.dgdv(v)

    C = f + g
    P = f * g
    grad_phi_square = 4 * x**2 + (2 * y - C)**2
    dC_dv = df_dv + dg_dv
    dP_dv = df_dv * g + f * dg_dv

    dphi_dv = dP_dv - dC_dv * y

    dx_dv = - 2*x * (dphi_dv / grad_phi_square)
    dy_dv = - (2*y - C) * (dphi_dv / grad_phi_square)

    return np.array([
        dx_dv,
        dy_dv,
    ])

def _logcosh(x):
    """
    log(cosh(x)), evaluated so that it cannot overflow.

    cosh(x) itself returns inf beyond |x| of about 710, and the ratio of two
    such gives inf or nan. That is reachable here: a transition width of 0.1 deg
    over a 90 deg span already puts the argument at 1100. This identity is exact
    and safe everywhere.
    """
    abs_x = np.abs(x)
    return abs_x + np.log1p(np.exp(-2.0*abs_x)) - np.log(2.0)


def beta(
    v: float,                  # [rad]
    v_transition: float,       # [rad]
    dv_transition_width: float,# [rad]
):
    """
    Base shape function for building f and g: a smooth, odd, monotonically
    increasing RAMP.

    It is the antiderivative of `dbeta_dv` that vanishes at the origin, so the
    two satisfy exactly

        d(beta)/dv  =  dbeta_dv

    Shape. beta rises with unit slope through the origin, then bends over once
    |v| passes v_transition and levels off at

        beta(+/- inf)  =  +/- v_transition / tanh(v_transition/dv_transition_width)

    which is +/- v_transition to within a fraction of a percent whenever the
    width is small compared with v_transition. So beta is in effect "v itself,
    smoothly clipped to +/- v_transition".

    Exact properties:

        beta(0)          =  0
        beta(-v)         = -beta(v)                 (odd)
        d(beta)/dv at 0  =  1                       (unit slope at the origin)

    Parameters
    ----------
    v                   : pseudo-latitude, in degree.
    v_transition        : where the ramp levels off, in degree. The two knees
                          sit at v = +v_transition and v = -v_transition.
    dv_transition_width : how abruptly it levels off, in degree. Small values
                          give a sharp corner, large values a gentle bend.

    Returns
    -------
    In the same units as v (degree), because integrating the dimensionless
    `dbeta_dv` over v carries v's units. Worth watching when combining beta with
    coefficients that feed f and g, which are dimensionless stereographic radii.
    """
    c = 2.0 * np.tanh(v_transition/dv_transition_width)
    return (
          _logcosh( (v + v_transition) / dv_transition_width )
        - _logcosh( (v - v_transition) / dv_transition_width )
    ) * dv_transition_width / c


def dbeta_dv(
    v: float,
    v_transition: float,
    dv_transition_width: float,
):
    """
    Derivative of `beta`: a smooth, even PLATEAU -- a top hat with rounded
    shoulders -- built from the difference of two hyperbolic tangents.

    Shape. Flat at 1 near v = 0, falling away through the two shoulders at
    v = +/- v_transition, and decaying to 0 beyond them.

    Exact properties:

        dbeta_dv(0)   =  1, and this is the maximum
        dbeta_dv(-v)  =  dbeta_dv(v)                (even)

        dbeta_dv(+/- v_transition)
                      =  tanh(2*v_transition/w) / (2*tanh(v_transition/w))

    where w = dv_transition_width. That last value is the HALF maximum, 0.5, to
    within a fraction of a percent whenever w is small compared with
    v_transition -- 0.50034 for v_transition = 20 deg, w = 5 deg. So
    v_transition marks the half-height points of the plateau, not its foot.

    The 1/c normalisation is what pins the peak at exactly 1; without it the
    amplitude would drift as the width is changed.

    Parameters
    ----------
    v                   : pseudo-latitude, in degree.
    v_transition        : half-width of the plateau, in degree.
    dv_transition_width : shoulder width, in degree.

    Returns
    -------
    Dimensionless, and independent of the units of v, being a ratio of tanh
    differences. Only `beta` carries units.
    """
    c = 2.0 * np.tanh(v_transition/dv_transition_width)
    return (
            np.tanh( (v + v_transition) / dv_transition_width )
          - np.tanh( (v - v_transition) / dv_transition_width )
    ) / c


class DisplacedPoleGrid:
    """
    Displaced Pole Grid

    Reference:

      - Madec, G. and M. Imbard (1996), A global ocean mesh to overcome the North
        Pole singularity. Climate Dynamics 12(6), 381-388.

    Mesh parallels (J curves) are circles in the north polar stereographic plane

        x^2 + y^2 - ( f + g ) y + f g = 0

        f = the northern crossing of the y-axis
        g = the southern crossing of the y-axis
        centre = ( 0, (f+g)/2 )        radius = (g-f)/2

    Pseudo-latitude:    v [deg]
    Pseudo-longitude:   u [deg]

    ---

    # The functions f and g, and how to solve them

    The function f is found by first defining its derivative, df/dv, with respect to pseudo
    latitude because the resolution of grid along the y-axis, degrees per grid, corresponds
    to scale factor along the y-axis. 

    In Madec and Imbard (1996) they use hyperbolic tangent to construct the resolution dg/dj.
    It is useful, but I alternate the way to prettify the math. First of all, dg/dj depends on
    the grid resolution, and g' = g'(j) is difficult when designing the function. Therefore,
    I enforce every expression to be in pseudo-latitude v. The only exception is that users
    provide resolution dlat/dj because that is intuitive. The dlat/dj will translate into dg/dv
    through the chain rule, as will be explained in later sections.

    The df/dv and dg/dv are designed as a sum of pleateau-shaped functions beta. 

        df/dv(v)   = A1 - W^f_trop * dbeta^f_trop/dv + W^f_polar * dbeta^f_polar/dv
        dg/dv(v<0) = -df/dv
        dg/dv(v>0) = -B1 - W^g_polar * dbeta^g_polar/dv

    where A1 and B1 are the base resolution, W is the amplitude of the resolution transition,
    with supscript being the function it belongs to and subscript being the location the transition
    happens. Because f' > 0 and g' < 0 are two necessary conditions, the signs are chosen so that
    we should expect W terms to be positive to be physically interpretable. The function dbeta/dv
    is made of two hyperbolic tangents,

        dbeta/dv(v) = c ( tanh((v+v_tran)/Delta_v) + tanh((v-v_tran)/Delta_v) )

    where v_tran is the transition latitude, Delta_v the transition width, and c the normalization
    constant such that dbeta/dv(v=0) = 1.

    The f and g follows

        f(v)   = A0 + A1 v - W^f_trop * beta^f_trop + W^f_polar * beta^f_polar
        g(v<0) = -f
        g(v>0) = B0 - B1 v - W^g_polar * beta^g_polar

    Because Claude points out that the paper's boundary conditions do not seem to satisfy the 
    condition that f(v=90deg) = g(v=90deg), I decide to do it in my way. So, the system I am 
    solving is

        Unknowns: A0, A1, B0, B1, W^f_trop, W^f_polar, and W^g_polar.
        Boundary conditions:
            
            (1) f(0) = -1         => A0 = -1
            (2) g(0) =  1         => B0 =  1
            (3) df/dv(v_trop) =   s_trop
            (4) dg/dv(v_trop) = - s_trop
            (5) f(90deg) = y_np
            (6) g(90deg) = y_np
            (7) f'(v_polar) = s_polar

    where s_0 and s_g is the resolution at the equator and the maximum latitude of the grid,
    and y_np being the y location of the displaced north pole.

    # How to see df/dv and dg/dv encompass grid resolution

    The scale factor along the meridional direction is

        e_2 = a sqrt( (dlon/dj * cos(lat))^2 + (dlat/dj)^2 )

    where j is the grid index. Evaluate the trajectory tracking the y-axis intercept
    (x, y) = (0, g(u)) gives

        e_2 = a dlat/dj
            = a d/dj (pi/2 - 2 arctan(|g|) )
            = -2 a sgn(g) / (1 + g^2) * (dg/dv) (dv/dj) 
    
    Rearranging the above gives

        dg/dv = -1/2 (dlat/dj) sgn(g) (1+g^2) / (dv/dj)

    and similarly for f
        
        df/dv = -1/2 (dlat/dj) sgn(f) (1+f^2) / (dv/dj)

    This is a pratical expression because user can specify the grid resolution dlat/dj
    and obtain df/dv for a particular location. 
    
    """
   
    def __init__(
        self,
        displaced_north_pole_lat: float, # [rad]
        v_trans_tropics_f: float,         # [rad]
        v_trans_width_tropics_f: float,   # [rad]
        v_trans_polar_f: float,          # [rad]
        v_trans_width_polar_f: float,    # [rad]
        v_trans_g: float,                # [rad]
        v_trans_g_width: float,          # [rad]
        resolution_equator: float,       # [rad/grid]
        resolution_polar: float,         # [rad/grid] the resolution of the point speficied by resolution_polar_v
        resolution_polar_v: float,       # [rad] location of a high latitude point
        number_of_rows: int,
        number_of_columns: int,
    ):
        self.displaced_north_pole_lat = displaced_north_pole_lat 
        self.v_trans_tropics_f = v_trans_tropics_f
        self.v_trans_width_tropics_f = v_trans_width_tropics_f
        self.v_trans_polar_f = v_trans_polar_f
        self.v_trans_width_polar_f = v_trans_width_polar_f
        self.v_trans_g = v_trans_g
        self.v_trans_g_width = v_trans_g_width
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
       
        self.v_bounds = np.linspace(-np.pi/2, np.pi/2, number_of_rows+1)
        self.dvdj = np.pi / number_of_rows
    
        self.solve_for_coefficients()

    def f(self, v):
        return -1.0 + self.A1 * v + (
            - self.W_tropics_f * beta(v, self.v_trans_tropics_f, self.v_trans_width_tropics_f)
            + self.W_polar_f * beta(v, self.v_trans_polar_f, self.v_trans_width_polar_f)
        )

    def dfdv(self, v):
        return self.A1 + (
            - self.W_tropics_f * dbeta_dv(v, self.v_trans_tropics_f, self.v_trans_width_tropics_f)
            + self.W_polar_f * dbeta_dv(v, self.v_trans_polar_f, self.v_trans_width_polar_f)
        )

    def dfdj(self, j):
        v = self.v_bounds[j]
        return self.dfdv(v) * self.dvdj(j)

    def g(self, v):
        if v < 0:
            return - self.f(v)
        else:
            return 1.0 - self.B1 * v - (
                self.W_g * beta(v, self.v_trans_g, self.v_trans_g_width)
            )

    def dgdv(self, v):
        if v < 0:
            return - self.dfdv(v)
        else:
            return - self.B1 - (
                self.W_g * dbeta_dv(v, self.v_trans_g, self.v_trans_g_width)
            )

    def grid_resolution_to_dgdv(
        self,
        grid_resolution: float, # [rad/grid]
        g: float,               # dimensionless
    ):
        
        return -0.5 * grid_resolution * np.sign(g) * (1 + g**2) / self.dvdj

    def grid_resolution_to_dfdv(
        self,
        grid_resolution: float, # [rad/grid]
        f: float,               # dimensionless
    ):
        # The way they are computed are identical        
        return self.grid_resolution_to_dgdv(grid_resolution, f)


    def solve_for_coefficients(self):
        
        s0 = self.grid_resolution_to_dfdv(self.resolution_equator, f=-1.0)
        
        # Solve for g's coefficients first
        dbetadv_g = dbeta_dv(0.0, self.v_trans_g, self.v_trans_g_width)
        beta_g = beta(np.pi/2, self.v_trans_g, self.v_trans_g_width)
        _, y_NP = spherical_to_stereo(np.pi/2, self.displaced_north_pole_lat)

        x = np.linalg.solve(
            np.array([
                [1,       dbetadv_g],
                [np.pi/2, beta_g]
            ]),
            np.array([
                s0,
                1 - y_NP,
            ])
        )

        self.B1 = x[0]
        self.W_g = x[1]


        # Then solve for f
                
              
 
    def generate_J_curve(
        self,
        v: float,
    ):
        """
        Compute the J curve (mesh zonal circle) given the mesh latitude s_j.
        """

        pts = np.zeros((2, self.number_of_columns))
        dh = 2*np.pi / self.number_of_columns
        
        h = 0.0 + np.arange(self.number_of_columns) * dh

        f = self.f(v)
        g = self.g(v)

        r   = (g - f) / 2.0
        y_c = (g + f) / 2.0

        x =       r * np.cos(h)
        y = y_c + r * np.sin(h)

        lon, lat = project_stereo_to_sphere(x, y)
        pts[0, :] = lon
        pts[1, :] = lat

        return pts

    def generate_I_curve(
        self,
        u: float,
        split: int = 5, 
    ):
        """
        Compute the I curve (mesh zonal circle) given the mesh longitude u.
        """

        _I_curve_ray_tracing_system = functools.partial(I_curve_ray_tracing_system, FG_funcs=self)

        # initial point: on the equator
        x0 = np.cos(u)
        y0 = np.sin(u)
        number_of_grids = self.number_of_rows_in_northern_hemisphere
        integration_steps = number_of_grids * split 
        dv = np.pi/2 / integration_steps
        
        _, pts_xy = integrate_euler_forward(_I_curve_ray_tracing_system, 0.0, [x0, y0], dv, integration_steps)

        lon, lat = project_stereo_to_sphere(pts_xy[0, ::split], pts_xy[1, ::split])
        pts = np.zeros((2, number_of_grids + 1))
        pts[0, :] = lon
        pts[1, :] = lat

        return pts

if __name__ == "__main__":

    displaced_pole_grid = DisplacedPoleGrid(
        v_trans_tropics_f = 20.0,         # [deg]
        v_trans_width_tropics_f = 10.0,   # [deg]
        amp_tropics_f = 1.04,             # [radius/grid_index]
        v_trans_polar_f = 70.0,          # [deg]
        v_trans_width_polar_f = 10.0,
        amp_polar_f = 1.0, # [radius/grid_index]
        v_trans_g: float, # [deg]
        v_trans_g_width: float, # [deg]
        amp_g: float, # [radius/grid_index]
        number_of_rows_in_northern_hemisphere: int,
        number_of_rows_in_southern_hemisphere: int,
        number_of_columns: int,
    )
    J_curves = [spherical_to_cartesian(generate_J_curve_from_s(s, 50)) for s in np.linspace(-np.pi/2, np.pi/2, 10)]
    I_curves = [spherical_to_cartesian(generate_I_curve(s_i, 20, 25, 1)) for s_i in np.linspace(0.0, 2*np.pi, 10)[::-1]]

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


