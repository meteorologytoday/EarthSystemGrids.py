import abc
import functools

import numpy as np
from scipy.optimize import brentq

from EarthSystemGrids.base import StructuredQuadMesh, apply_ocean_mask

# The northern branch runs from the equator to the mesh north pole, and the
# pseudo-latitude v mirrors geographic latitude, so the pole sits at v = pi/2.
# This is structural, not a convention that can be varied: the grid discretises
# [0, V_MAX] into number_of_rows_in_NH rows, and a formulation that met its
# boundary conditions at some other v_max would meet them at a location the grid
# never visits.
V_MAX = np.pi / 2

def spherical_to_stereo(lon:float, lat:float):
    """
    Project sphere coordinate to stereographic.
    Unit circle means equator, and center means northpole.
    """
    rho = np.tan(np.pi/4 - lat/2)
    x = rho * np.cos(lon)
    y = rho * np.sin(lon)
    return x, y


def stereo_to_spherical(x:float, y:float):
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

def integrate_euler_forward(dydt, t0, y0, dt, steps, substeps: int = 1):
    """
    Forward Euler that records only every `substeps`-th point.

    Each output interval `dt` is covered by `substeps` Euler steps of
    `dt/substeps`, and the intermediate points are discarded. This decouples
    accuracy from the spacing of the points you want back: `steps` sets how many
    values are returned, `substeps` sets how hard the integrator works between
    them. Useful near the mesh pole, where the ODE stiffens as the J-curve
    radius shrinks while the rows are still widely spaced in v.

    Forward Euler is first order, so the error falls only linearly in
    `substeps`; `integrate_rk4` reaches the same accuracy with far fewer
    right-hand-side evaluations.

    Returns
    -------
    t : (steps+1,)
    y : (dim, steps+1)
    """
    dim = len(y0)
    h = dt / substeps
    y = np.zeros((dim, steps+1))
    t = np.zeros((steps+1,))
    t[0] = t0
    y[:, 0] = y0

    y_now = np.array(y0, dtype=float)
    t_now = float(t0)
    for step in range(steps):

        for _ in range(substeps):
            y_now = y_now + dydt(t_now, y_now) * h
            t_now = t_now + h

        t_now = t0 + (step + 1) * dt      # remove accumulated drift
        t[step+1] = t_now
        y[:, step+1] = y_now

    return t, y

def integrate_rk4(dydt, t0, y0, dt, steps, substeps: int = 1):
    """
    Classical fourth-order Runge-Kutta, with the same interface as
    `integrate_euler_forward`: `steps` output intervals of width `dt`, each
    covered by `substeps` internal steps whose intermediate values are
    discarded.

    Being fourth order, the error falls as substeps^-4 rather than the Euler
    substeps^-1, so it reaches a given accuracy with far fewer right-hand-side
    evaluations. That matters near the mesh pole, where the J-curve radius
    shrinks and the ODE stiffens while the rows are still widely spaced in v.

    Returns
    -------
    t : (steps+1,)
    y : (dim, steps+1)
    """
    dim = len(y0)
    h = dt / substeps
    out = np.zeros((dim, steps+1))
    t = np.zeros((steps+1,))
    t[0] = t0
    out[:, 0] = y0

    y_now = np.array(y0, dtype=float)
    t_now = float(t0)
    for step in range(steps):

        for _ in range(substeps):
            k1 = dydt(t_now,         y_now)
            k2 = dydt(t_now + h/2.0, y_now + h/2.0*k1)
            k3 = dydt(t_now + h/2.0, y_now + h/2.0*k2)
            k4 = dydt(t_now + h,     y_now + h*k3)
            y_now = y_now + h/6.0*(k1 + 2*k2 + 2*k3 + k4)
            t_now = t_now + h

        t_now = t0 + (step + 1) * dt      # remove accumulated drift
        t[step+1] = t_now
        out[:, step+1] = y_now

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


class FGBase(abc.ABC):
    """
    A formulation of the two y-axis crossings f(v) and g(v) of the mesh parallels.

    The J-curve labelled by the pseudo-latitude v is the circle

        x^2 + y^2 - ( f(v) + g(v) ) y + f(v) g(v) = 0

    so a formulation is completely described by f and g and their derivatives.
    DisplacedPoleGrid consumes nothing else: it never reads a coefficient, a
    transition latitude, or a boundary condition. Everything below the four
    abstract methods is the formulation's own business, including how it solves
    for its coefficients and how it checks them.

    Resolution independence
    -----------------------
    Resolutions are given per unit v, never per grid row, so a formulation never
    sees dv/dj or the row count. The shape is a fixed curve and dv/dj carries the
    resolution; changing the number of rows must not change this object.

    v runs over [0, V_MAX] with V_MAX = pi/2, the mesh north pole. That end
    point is fixed by the construction, so it is not a constructor argument;
    `self.v_max` exists only so the conditions read naturally.

    Subclasses must implement f, dfdv, g, dgdv, and may override
    `_scheme_checks` and `transition_marks`.
    """

    def __init__(self, displaced_north_pole_lat, dlat_dv_equator, ring_radius=0.0):
        self.displaced_north_pole_lat = displaced_north_pole_lat
        self.v_max = V_MAX
        _, self.y_NP = spherical_to_stereo(np.pi/2, displaced_north_pole_lat)

        # Madec and Imbard's condition (c) closes the mesh pole exactly,
        # f(v_max) = g(v_max), which leaves the last J-curve a single point:
        # number_of_columns coincident cells, an undefined e1, and |grad phi|^2
        # -> 0 so the I-curve ODE is 0/0 there. The paper does not impose (c)
        # either -- it truncates and notes the remainder is on land. ring_radius
        # is that truncation: the last J-curve keeps radius R about the pole.
        #
        # It costs headroom. Opening the ring raises g_target, so g has less
        # distance to cover while C1 still forces it to start at s0, and the
        # limit s0 < 2(1 - g_target)/v_max tightens by roughly 1.3 R.
        self.ring_radius = ring_radius
        self.f_target = self.y_NP - ring_radius
        self.g_target = self.y_NP + ring_radius
        if ring_radius >= abs(self.y_NP):
            raise ValueError(
                f"ring_radius {ring_radius:.4f} must stay below |y_NP| = "
                f"{abs(self.y_NP):.4f}, otherwise the hole swallows the origin "
                f"and the GEOGRAPHIC north pole falls outside the mesh.")
        # At the equator f = -1, so the Jacobian 2/(1+f^2) is exactly 1 and this
        # reduces to s0 = dlat_dv_equator. Written through the general
        # conversion anyway, so there is only one code path.
        self.s0 = self.dlat_dv_to_dwdv(dlat_dv_equator, -1.0)

    # ---- the interface DisplacedPoleGrid depends on --------------------------

    @abc.abstractmethod
    def f(self, v):
        """Crossing of the J-curve with the -y axis (longitude 270)."""

    @abc.abstractmethod
    def dfdv(self, v):
        """df/dv."""

    @abc.abstractmethod
    def g(self, v):
        """Crossing of the J-curve with the +y axis (longitude 90)."""

    @abc.abstractmethod
    def dgdv(self, v):
        """dg/dv."""

    # ---- shared helpers ------------------------------------------------------

    @staticmethod
    def dlat_dv_to_dwdv(dlat_dv, w):
        """
        Convert a latitude rate into an ordinate rate at crossing value w.

        The stereographic radius is rho = tan(pi/4 - lat/2), so
        drho/dlat = -(1/2)(1 + rho^2). The sign of w selects the branch: on the
        f side w < 0 and the result is positive (f increases), on the g side
        w > 0 and it is negative (g decreases).

        Note this degenerates at w = 0, where the crossing passes over the
        geographic pole and the folded latitude has a maximum. Evaluate it only
        at points where the branch is unambiguous.
        """
        return -0.5 * dlat_dv * np.sign(w) * (1 + w**2)

    @staticmethod
    def dlat_dv_to_dfdv(dlat_dv, f):
        """
        Latitude rate -> df/dv on the f branch, using the CONTINUED latitude
        psi_f = pi/2 + 2 arctan(f), which increases monotonically as the
        crossing travels north over the geographic pole and down the far side.

        Always positive, and smooth through f = 0. Prefer this to
        `dlat_dv_to_dwdv` whenever the evaluation point may lie past the
        crossing, where the folded latitude has a maximum and sign(f) flips.
        """
        return +0.5 * dlat_dv * (1 + f**2)

    @staticmethod
    def dlat_dv_to_dgdv(dlat_dv, g):
        """
        As above on the g branch, psi_g = pi/2 - 2 arctan(g). Always negative.
        g stays positive throughout, so this branch never folds.
        """
        return -0.5 * dlat_dv * (1 + g**2)

    def transition_marks(self):
        """
        [(v, label)] of interesting locations, for diagnostic plots. Lets a plot
        annotate a formulation without knowing what kind it is. Default: none.
        """
        return []

    def _scheme_checks(self):
        """
        Extra verification rows specific to this formulation, as tuples
        (name, measured, expected, kind) with kind in {eq, eq_loose, gt, lt}.
        Default: none.
        """
        return []

    # ---- verification --------------------------------------------------------

    def verify(self, tolerance: float = 1e-9, samples: int = 2001,
               verbose: bool = True, raise_on_failure: bool = False):
        """
        Check that this formulation does what it claims.

        Two groups are tested, and they fail for different reasons:

        1. Boundary conditions. For a formulation that solves a linear system
           these are outputs of that solve and should hold to machine precision;
           a failure means the system was assembled wrongly.

        2. Structural properties, which are NOT automatic. Satisfying every
           boundary condition and still producing an unusable grid is the normal
           failure mode: f can fold back, the circles can stop being embedded,
           or the pole can miss its target. These are properties (a)-(e) of
           Madec and Imbard (1996).

        Only v in [0, v_max] is examined. Anything a formulation does at
        negative v is its own affair; the southern hemisphere of the grid is a
        lat-lon patch that does not use f or g.

        Returns True if every check passes.
        """
        v = np.linspace(0.0, self.v_max, samples)
        f_v    = np.array([self.f(vv)    for vv in v])
        g_v    = np.array([self.g(vv)    for vv in v])
        dfdv_v = np.array([self.dfdv(vv) for vv in v])
        dgdv_v = np.array([self.dgdv(vv) for vv in v])
        radius = (g_v - f_v) / 2.0

        pole_lat = np.rad2deg(np.pi/2 - 2*np.arctan(np.abs((f_v[-1] + g_v[-1])/2.0)))

        checks = [
            # name                          measured                          expected   kind
            ("f(0) = -1",                   self.f(0.0),                      -1.0,      "eq"),
            ("g(0) = +1",                   self.g(0.0),                      +1.0,      "eq"),
            ("dfdv(0) = s0",                self.dfdv(0.0),                   self.s0,   "eq"),
            ("dgdv(0) = -s0",               self.dgdv(0.0),                  -self.s0,   "eq"),
            ("f(v_max) = f_target",         self.f(self.v_max),          self.f_target, "eq"),
            ("g(v_max) = g_target",         self.g(self.v_max),          self.g_target, "eq"),
            ("last ring radius = R",        (g_v[-1] - f_v[-1])/2.0,  self.ring_radius, "eq"),
            ("C1 at equator: f'+g' = 0",    self.dfdv(0.0) + self.dgdv(0.0),   0.0,      "eq"),
            ("pole latitude [deg]",         pole_lat,
             np.rad2deg(self.displaced_north_pole_lat),                                  "eq_loose"),
            ("(b) f strictly increasing",   dfdv_v.min(),                      0.0,      "gt"),
            ("(b) g strictly decreasing",   dgdv_v.max(),                      0.0,      "lt"),
            ("(a) radius > 0 for v < v_max", radius[:-1].min(),                0.0,      "gt"),
        ] + list(self._scheme_checks())

        ok = True
        rows = []
        for name, got, want, kind in checks:
            if kind == "eq":
                passed = abs(got - want) <= tolerance
                detail = f"{got:+14.9f}  want {want:+14.9f}  err {abs(got-want):.1e}"
            elif kind == "eq_loose":
                passed = abs(got - want) <= 1e-6 * max(1.0, abs(want))
                detail = f"{got:+14.9f}  want {want:+14.9f}  err {abs(got-want):.1e}"
            elif kind == "gt":
                passed = got > 0.0
                detail = f"min {got:+14.9f}  must be > 0"
            else:
                passed = got < 0.0
                detail = f"max {got:+14.9f}  must be < 0"
            ok = ok and passed
            rows.append((passed, name, detail))

        if verbose:
            print(f"{type(self).__name__}.verify:")
            for passed, name, detail in rows:
                print(f"   [{'ok' if passed else 'FAIL'}] {name:30s} {detail}")
            print(f"   -> {'all checks passed' if ok else 'FAILURES ABOVE'}")

        if raise_on_failure and not ok:
            failed = [name for passed, name, _ in rows if not passed]
            raise AssertionError(f"{type(self).__name__}.verify failed: "
                                 + ", ".join(failed))
        return ok


class FGMI1996(FGBase):
    """
    The formulation of Madec and Imbard (1996): a linear term plus smooth
    log-cosh ramps, with the coefficients found from a linear system.

        f(v) = -1 + A1 v - W_tropics_f beta_tropics(v) + W_polar_f beta_polar(v)
        g(v) = +1 - B1 v - W_g beta_g(v)

    where beta is the ramp defined above. The paper builds f' and g' from
    hyperbolic tangents "so that the grid spacing can be rather easily adjusted
    to the desired values"; the transition parameters are what buys that local
    control, and are the price of it relative to a closed-form scheme.

    Boundary conditions
    -------------------
        f(0) = -1                    g(0) = +1
        dfdv(0) = s0                 dgdv(0) = -s0        (C1 across the equator)
        dfdv(v_polar) = s_polar      g(v_max) = y_NP
        f(v_max) = y_NP

    Three on g (2 unknowns B1, W_g after g(0) is hardcoded), four on f
    (3 unknowns A1, W_polar_f, W_tropics_f after f(0) is hardcoded).

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

    def __init__(self,
                 displaced_north_pole_lat,      # [rad]
                 dlat_dv_equator,               # [rad lat / rad v]
                 dlat_dv_polar,                 # [rad lat / rad v] at v_polar
                 v_polar,                       # [rad] where dlat_dv_polar applies
                 v_trans_tropics_f,             # [rad]
                 v_trans_width_tropics_f,       # [rad]
                 v_trans_polar_f,               # [rad]
                 v_trans_width_polar_f,         # [rad]
                 v_trans_g,                     # [rad]
                 v_trans_g_width,               # [rad]
                 ring_radius=0.0):              # [dimensionless] last-ring radius
        super().__init__(displaced_north_pole_lat, dlat_dv_equator, ring_radius)
        self.dlat_dv_polar = dlat_dv_polar
        self.v_polar = v_polar
        self.v_trans_tropics_f = v_trans_tropics_f
        self.v_trans_width_tropics_f = v_trans_width_tropics_f
        self.v_trans_polar_f = v_trans_polar_f
        self.v_trans_width_polar_f = v_trans_width_polar_f
        self.v_trans_g = v_trans_g
        self.v_trans_g_width = v_trans_g_width

        self._solve()
        if not self.verify(verbose=False):
            print(f"{type(self).__name__}: verify() reported failures; "
                  f"call verify() for the detail.")

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

    def transition_marks(self):
        return [
            (self.v_trans_tropics_f, "v_trans_tropics_f"),
            (self.v_trans_polar_f,   "v_trans_polar_f"),
            (self.v_trans_g,         "v_trans_g"),
            (self.v_polar,           "v_polar"),
        ]

    c_f_const = -1.0   # f(0), hardcoded in f()
    c_g_const = +1.0   # g(0), hardcoded in g()

    def _solve(self):
        v_star = -np.abs(self.v_polar)
        _, y_star = spherical_to_stereo(-np.pi/2, v_star)

        s0     = self.s0
        s_star = self.dlat_dv_to_dwdv(self.dlat_dv_polar, y_star)
        self._s_star = s_star
        self._v_star = v_star

        # g: 2 unknowns from dgdv(0) = -s0 and g(v_max) = y_NP
        self.B1, self.W_g = np.linalg.solve(
            np.array([
                [1,          dbeta_dv(0.0, self.v_trans_g, self.v_trans_g_width)],
                [self.v_max, beta(self.v_max, self.v_trans_g, self.v_trans_g_width)],
            ]),
            np.array([s0, self.c_g_const - self.g_target]),
        )

        # f: 3 unknowns from dfdv(0) = s0, dfdv(v_star) = s_star, f(v_max) = y_NP.
        # The third is a VALUE condition, so its amplitude entries are beta, not
        # dbeta_dv -- the single easiest thing to get wrong here.
        tt, wt = self.v_trans_tropics_f, self.v_trans_width_tropics_f
        tp, wp = self.v_trans_polar_f,   self.v_trans_width_polar_f
        self.A1, self.W_polar_f, self.W_tropics_f = np.linalg.solve(
            np.array([
                [1,          dbeta_dv(0.0,       tp, wp), -dbeta_dv(0.0,       tt, wt)],
                [1,          dbeta_dv(v_star,    tp, wp), -dbeta_dv(v_star,    tt, wt)],
                [self.v_max, beta(self.v_max,    tp, wp), -beta(self.v_max,    tt, wt)],
            ]),
            np.array([s0, s_star, self.f_target - self.c_f_const]),
        )

    def _scheme_checks(self):
        return [
            ("dfdv(v_polar) = s_polar", self.dfdv(self._v_star), self._s_star, "eq"),
            ("design: W_polar_f   > 0", self.W_polar_f,          0.0,          "gt"),
            ("design: W_tropics_f > 0", self.W_tropics_f,        0.0,          "gt"),
            ("design: W_g         > 0", self.W_g,                0.0,          "gt"),
        ]


class FGPolynomialStep(FGBase):
    """
    Smoothstep formulation: the RATE is a cubic step between two levels.

        gamma(t) = 1 - 3 t^2 + 2 t^3          gamma(0)=1, gamma(1)=0
        f'(v)    = B + A gamma(v / w)
        f (v)    = c + B v + A w Gamma(v / w),   Gamma(t) = t - t^3 + t^4/2

    Three unknowns per branch and three conditions, so the solve is a 2x2 after
    the constant falls out. `w` is a hand-chosen shape parameter, not solved.

    Why w >= V_MAX matters
    ----------------------
    gamma is monotone decreasing only on [0, 1]; past t = 1 its derivative
    6t(t-1) turns positive and it grows again. Requiring w >= V_MAX keeps the
    argument inside [0, 1], and then f' is monotone in v and therefore trapped
    between its own endpoint values. Monotonicity of f collapses to a single
    scalar test that can be made before evaluating anything:

        f'(0) = s0 > 0 by construction, so  f increasing  <=>  f'(v_max) > 0

    The determinant of the 2x2 is w Gamma(T) - v_max = -v_max T^2 (1 - T/2)
    with T = v_max/w, strictly negative for T in (0, 1]. It vanishes only as
    w -> infinity, where gamma is flat and f' cannot bend enough to reach y_NP.

    Boundary conditions
    -------------------
        f(0) = -1,  f'(0) = +s0,  f(v_max) = y_NP
        g(0) = +1,  g'(0) = -s0,  g(v_max) = y_NP

    The polar rates f'(v_max) and g'(v_max) are OUTCOMES, not inputs. That is
    the price of the simple solve and the cheap guarantee.

    The g branch is the binding one
    ------------------------------
    f travels y_NP + 1 while g travels only y_NP - 1 -- 2.75x less at a 40 N
    mesh pole -- yet C1 across the equator forces both to start at |slope| = s0.
    So g must shed much more slope over the same v range, and it is g, not f,
    that turns around first. At w = V_MAX the condition is exactly

        s0 < 2 (1 - y_NP) / v_max

    which is 0.68 for a pole at 40 N and only 0.38 at 20 N. In grid terms s0 is
    the equatorial spacing divided by dv/dj, so this says the equator must be
    meaningfully finer than uniform -- ordinary tropical refinement, but it is a
    real constraint and `_scheme_checks` guards both branches.
    """

    def __init__(self,
                 displaced_north_pole_lat,      # [rad]
                 dlat_dv_equator,               # [rad lat / rad v]
                 v_width_f,                     # [rad] must be >= V_MAX
                 v_width_g=None,                # [rad] defaults to v_width_f
                 ring_radius=0.0):              # [dimensionless] last-ring radius
        super().__init__(displaced_north_pole_lat, dlat_dv_equator, ring_radius)
        self.v_width_f = v_width_f
        self.v_width_g = v_width_f if v_width_g is None else v_width_g
        for name, w in (("v_width_f", self.v_width_f), ("v_width_g", self.v_width_g)):
            if w < self.v_max:
                raise ValueError(
                    f"{name} = {np.rad2deg(w):.2f} deg is below V_MAX = "
                    f"{np.rad2deg(self.v_max):.2f} deg. gamma is only monotone on "
                    f"[0, 1], so a narrower width lets the rate turn around inside "
                    f"the domain and the monotonicity guarantee is lost.")
        self._solve()
        if not self.verify(verbose=False):
            print(f"{type(self).__name__}: verify() reported failures; "
                  f"call verify() for the detail.")

    @staticmethod
    def _gamma(t):
        return 1.0 - 3.0*t**2 + 2.0*t**3

    @staticmethod
    def _Gamma(t):
        """Antiderivative of _gamma vanishing at 0."""
        return t - t**3 + 0.5*t**4

    def _branch(self, slope_at_0, total_rise, w):
        """(B, A) from  B + A = slope_at_0  and  B v_max + A w Gamma(T) = total_rise."""
        T = self.v_max / w
        return np.linalg.solve(
            np.array([[1.0,        1.0],
                      [self.v_max, w*self._Gamma(T)]]),
            np.array([slope_at_0, total_rise]))

    def _solve(self):
        self.c_f = -1.0
        self.c_g = +1.0
        self.B_f, self.A_f = self._branch(+self.s0, self.f_target - self.c_f, self.v_width_f)
        self.B_g, self.A_g = self._branch(-self.s0, self.g_target - self.c_g, self.v_width_g)

    def f(self, v):
        w = self.v_width_f
        return self.c_f + self.B_f*v + self.A_f*w*self._Gamma(v/w)

    def dfdv(self, v):
        return self.B_f + self.A_f*self._gamma(v/self.v_width_f)

    def g(self, v):
        if v < 0:
            return - self.f(v)
        w = self.v_width_g
        return self.c_g + self.B_g*v + self.A_g*w*self._Gamma(v/w)

    def dgdv(self, v):
        if v < 0:
            return - self.dfdv(v)
        return self.B_g + self.A_g*self._gamma(v/self.v_width_g)

    def max_dlat_dv_equator(self):
        """
        Largest s0 for which BOTH branches stay monotone, found from the closed
        form of the endpoint rates. Useful for reporting why a configuration was
        rejected, since g binds well before f does.
        """
        out = []
        for sign, c, w, tgt in ((+1.0, self.c_f, self.v_width_f, self.f_target),
                                (-1.0, self.c_g, self.v_width_g, self.g_target)):
            T = self.v_max / w
            # endpoint rate is affine in s0; solve  rate(s0) = 0  from two samples
            r = []
            for s in (1.0, 2.0):
                B, A = self._branch(sign*s, tgt - c, w)
                r.append(sign*(B + A*self._gamma(T)))
            out.append(1.0 + (2.0 - 1.0)*(0.0 - r[0])/(r[1] - r[0]))
        return min(out)

    def transition_marks(self):
        marks = [(self.v_width_f, "v_width_f")]
        if self.v_width_g != self.v_width_f:
            marks.append((self.v_width_g, "v_width_g"))
        return marks

    def _scheme_checks(self):
        """
        Both endpoint rates, which is the whole of the monotonicity question for
        this scheme: gamma keeps f' and g' trapped between their endpoints, so
        signs at the two ends are necessary and sufficient.
        """
        Tf = self.v_max / self.v_width_f
        Tg = self.v_max / self.v_width_g
        return [
            ("f'(v_max) > 0 (outcome)", self.B_f + self.A_f*self._gamma(Tf), 0.0, "gt"),
            ("g'(v_max) < 0 (outcome)", self.B_g + self.A_g*self._gamma(Tg), 0.0, "lt"),
        ]

class FGLogCosh(FGBase):
    """
    An adaptation of Madec and Imbard (1996): define the RATE as a constant plus
    a single hyperbolic tangent, so it is a smooth step between two levels.

        f'(v) =   C1_f - W_f tanh( (v - v_c_f) / delta_f )
        g'(v) = - C1_g + W_g tanh( (v - v_c_g) / delta_g )

    Integrating (the antiderivative of tanh(x/d) is d logcosh(x/d), NOT tanh):

        f(v) = C0_f + C1_f v - W_f delta_f logcosh( (v - v_c_f) / delta_f )
        g(v) = C0_g - C1_g v + W_g delta_g logcosh( (v - v_c_g) / delta_g )

    Boundary conditions, four per branch
    -----------------------------------
        f(0) = -1                  g(0) = +1
        dfdv(0) = s0               dgdv(0) = -s0        (C1 across the equator)
        f(v_max) = f_target        g(v_max) = g_target
        dfdv(v_max) = s_f          dgdv(v_max) = s_g

    The widths delta are given, so each branch has four unknowns -- C0, C1, W and
    v_c -- matching its four conditions.

    Why this scheme is worth the root-find
    -------------------------------------
    tanh is monotone on the WHOLE line, so f' is monotone in v and therefore
    trapped between its two endpoint values -- and here both endpoints are
    inputs. Hence

        f strictly increasing  <=>  s0 > 0 and s_f > 0
        g strictly decreasing  <=>  s0 > 0 and s_g < 0

    guaranteed by construction, with nothing to sample and no region test.

    More importantly, <f'> is NOT pinned to (f'(0) + f'(v_max))/2, because v_c is
    free to move. FGLinear and FGPolynomialStep are both locked to that average
    -- trivially for a straight line, and for gamma because it integrates to 1/2
    over [0, 1] -- which forces g'(v_max) = 2<g'> - g'(0) and makes g reverse
    once s0 exceeds 2(1 - g_target)/v_max. This scheme escapes that entirely, so
    it accepts equatorial resolutions the others reject.

    Solving
    -------
    The two slope conditions give W and C1 in closed form for any v_c, and the
    two value conditions collapse to a single scalar equation for v_c, solved by
    bisection. Fixing the width and solving for the location (rather than the
    reverse) matters: v_c is a LOCATION, so a few widths either side of
    [0, v_max] brackets every case, whereas a width is a SCALE needing a
    geometric sweep with no natural upper cut -- and a root found at an absurd
    width satisfies the conditions while making the tanh linear across the whole
    domain, silently destroying the shape that was asked for. Measured over a
    sweep of (s0, s_polar), solving for v_c found a usable answer roughly three
    times as often.

    A v_c outside [0, v_max] is normal and healthy: it just means the knee sits
    beyond the domain and only its tail is used.
    """

    def __init__(self,
                 displaced_north_pole_lat,      # [rad]
                 dlat_dv_equator,               # [rad lat / rad v]
                 dlat_dv_polar_f,               # [rad lat / rad v] at v_max, f branch
                 dlat_dv_polar_g,               # [rad lat / rad v] at v_max, g branch
                 v_width_f,                     # [rad] delta_f, given
                 v_width_g=None,                # [rad] delta_g, defaults to v_width_f
                 ring_radius=0.0):              # [dimensionless] last-ring radius
        super().__init__(displaced_north_pole_lat, dlat_dv_equator, ring_radius)
        self.v_width_f = v_width_f
        self.v_width_g = v_width_f if v_width_g is None else v_width_g

        # both targets lie past the f-branch zero crossing, so use the
        # continued-latitude Jacobians rather than the folded sign() form
        self.s_f = self.dlat_dv_to_dfdv(dlat_dv_polar_f, self.f_target)
        self.s_g = self.dlat_dv_to_dgdv(dlat_dv_polar_g, self.g_target)

        self._solve()
        if not self.verify(verbose=False):
            print(f"{type(self).__name__}: verify() reported failures; "
                  f"call verify() for the detail.")

    # ---- one branch ---------------------------------------------------------

    def _branch(self, value_0, value_max, slope_0, slope_max, w, sign):
        """
        Solve one branch. `sign` is +1 for f and -1 for g, matching the sign
        conventions in the class docstring, so that W comes out positive for the
        usual case of a rate that falls toward the pole.

        Returns (C0, C1, W, v_c).
        """
        V = self.v_max

        def closed_form(v_c):
            t0 = np.tanh(-v_c/w)
            tV = np.tanh((V - v_c)/w)
            if abs(tV - t0) < 1e-13:
                return None                      # both tails saturated, W blows up
            W  = (sign*slope_0 - sign*slope_max) / (tV - t0)
            C1 = sign*slope_0 + W*t0
            return W, C1

        def residual(v_c):
            r = closed_form(v_c)
            if r is None:
                return np.nan
            W, C1 = r
            L0 = _logcosh(-v_c/w)
            LV = _logcosh((V - v_c)/w)
            # value_max - value_0 = sign*C1*V - sign*W*w*(LV - L0)
            return sign*C1*V - sign*W*w*(LV - L0) - (value_max - value_0)

        # v_c is a LOCATION, so a few widths either side of the domain brackets it
        grid = np.linspace(-4.0*w, V + 4.0*w, 4001)
        vals = np.array([residual(x) for x in grid])
        ok = np.isfinite(vals)
        cross = np.where(ok[:-1] & ok[1:] &
                         (np.sign(vals[:-1])*np.sign(vals[1:]) < 0))[0]
        if len(cross) == 0:
            raise ValueError(
                "FGLogCosh: no v_c in [-4 delta, v_max + 4 delta] satisfies the two "
                "value conditions for this combination of rates and width. A "
                "narrower delta, or endpoint rates closer to the mean the "
                "geometry demands, will usually admit a solution.")
        v_c = brentq(residual, grid[cross[0]], grid[cross[0]+1], xtol=1e-15)
        W, C1 = closed_form(v_c)
        C0 = value_0 + sign*W*w*_logcosh(-v_c/w)
        return C0, C1, W, v_c

    def _solve(self):
        self.C0_f, self.C1_f, self.W_f, self.v_c_f = self._branch(
            -1.0, self.f_target, +self.s0, self.s_f, self.v_width_f, +1.0)
        self.C0_g, self.C1_g, self.W_g, self.v_c_g = self._branch(
            +1.0, self.g_target, -self.s0, self.s_g, self.v_width_g, -1.0)

    # ---- the interface ------------------------------------------------------

    def f(self, v):
        w = self.v_width_f
        return (self.C0_f + self.C1_f*v
                - self.W_f*w*_logcosh((v - self.v_c_f)/w))

    def dfdv(self, v):
        return self.C1_f - self.W_f*np.tanh((v - self.v_c_f)/self.v_width_f)

    def g(self, v):
        if v < 0:
            return - self.f(v)
        w = self.v_width_g
        return (self.C0_g - self.C1_g*v
                + self.W_g*w*_logcosh((v - self.v_c_g)/w))

    def dgdv(self, v):
        if v < 0:
            return - self.dfdv(v)
        return -self.C1_g + self.W_g*np.tanh((v - self.v_c_g)/self.v_width_g)

    def transition_marks(self):
        marks = [(self.v_c_f, "v_c_f")]
        if abs(self.v_c_g - self.v_c_f) > 1e-12:
            marks.append((self.v_c_g, "v_c_g"))
        return marks

    def _scheme_checks(self):
        """
        Monotonicity needs no test of its own: tanh keeps each rate trapped
        between its endpoints, and both endpoints are imposed. What is worth
        reporting is the solved knee locations, since a v_c far outside the
        domain means the tanh is running as a near-straight line and the width
        has stopped meaning anything.
        """
        V = self.v_max
        return [
            ("dfdv(v_max) = s_f",  self.dfdv(V), self.s_f, "eq"),
            ("dgdv(v_max) = s_g",  self.dgdv(V), self.s_g, "eq"),
            ("v_c_f within 4 delta of domain",
             4.0*self.v_width_f - abs(self.v_c_f - 0.5*V) + 0.5*V, 0.0, "gt"),
            ("v_c_g within 4 delta of domain",
             4.0*self.v_width_g - abs(self.v_c_g - 0.5*V) + 0.5*V, 0.0, "gt"),
        ]


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

    The formulation of f and g lives in a separate object (see `FGBase`), so a
    grid can be built from any scheme that provides f, dfdv, g and dgdv. This
    class asks for nothing else: no coefficient, no transition latitude, no
    boundary condition.

    The southern hemisphere is a plain lat-lon patch placed directly by
    `latitude_bounds_in_SH`, and does not use f or g at all. Only the northern
    branch, v in [0, pi/2], is parameterised by v.
    """
   
    def __init__(
        self,
        fg: FGBase,                      # the f/g formulation
        number_of_rows_in_NH: int,
        latitude_bounds_in_SH,           # An array of latitudes [rad], ascending, ending at 0
        number_of_columns: int,
        pole_longitude: float = np.pi/2,  # [rad] where to put the mesh pole
    ):
        self.fg = fg
        self.number_of_rows_in_NH = number_of_rows_in_NH
        self.latitude_bounds_in_SH = np.asarray(latitude_bounds_in_SH, dtype=float)
        self.number_of_columns = number_of_columns

        # The construction always builds the mesh pole on the +y axis of the
        # stereographic plane, i.e. at longitude 90 E, because g is by definition
        # the crossing on that side. Rotating the whole mesh about the Earth's
        # rotation axis moves it anywhere on that parallel, and costs nothing:
        # a rotation about the axis is a rotation of the stereographic plane
        # about its origin, which maps circles to circles and preserves angles,
        # so every property of the construction survives untouched. It is purely
        # a relabelling of longitude applied on the way out.
        self.pole_longitude = pole_longitude
        self._lon_offset = pole_longitude - np.pi/2

        # v parameterises only the northern branch, equator to mesh pole. dv/dj
        # carries the resolution and `fg` never sees it, which is what keeps the
        # formulation independent of how finely the grid is discretised.
        self.dvdj = V_MAX / number_of_rows_in_NH

    def _stereo_to_spherical(self, x, y):
        """
        stereo_to_spherical, then rotate longitude so the mesh pole sits at
        `pole_longitude` rather than at the construction's native 90 E.
        """
        lon, lat = stereo_to_spherical(x, y)
        if self._lon_offset != 0.0:
            lon = (lon + self._lon_offset + np.pi) % (2.0*np.pi) - np.pi
        return lon, lat

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

        if v >= 0:

            f = self.fg.f(v)
            g = self.fg.g(v)

            r   = (g - f) / 2.0
            y_c = (g + f) / 2.0
            
            x =       r * np.cos(h)
            y = y_c + r * np.sin(h)

            lon, lat = self._stereo_to_spherical(x, y)

        else:
            
            lat = v
            lon = h

        pts[0, :] = lon
        pts[1, :] = lat

        return pts

    def generate_I_curve(
        self,
        u: float,
        split: int = 10, 
    ):
        """
        Compute the I curve (mesh zonal circle) given the mesh longitude u.
        """

        _I_curve_ray_tracing_system = functools.partial(I_curve_ray_tracing_system, FG_funcs=self.fg)

        # initial point: on the equator
        x0 = np.cos(u)
        y0 = np.sin(u)
        # The mesh north pole is at v = pi/2, where f = g and the J-curve
        # degenerates to a point: |grad phi|^2 -> 0 and the ODE is 0/0 there.
        # Stop one row short so the integration never reaches it.
        number_of_grids = self.number_of_rows_in_NH - 1
        _, pts_xy = integrate_euler_forward(_I_curve_ray_tracing_system, 0.0, [x0, y0],
                                            self.dvdj, number_of_grids, substeps=split)

        lon, lat = self._stereo_to_spherical(pts_xy[0, :], pts_xy[1, :])
        pts = np.zeros((2, number_of_grids + 1))
        pts[0, :] = lon
        pts[1, :] = lat

        return pts

    def generate_mesh(self, latitude_bounds_in_SH=None, v_interfaces_in_NH=None,
                      split: int = 5, earth_radius: float = 6371.0e3):
        """
        Assemble the full cell mesh: centres, corners, scale factors and areas.

        Points are computed on a doubled grid, so that row/column interfaces
        sit at even indices and cell centres at odd ones. Cell (j, i) then has
        its centre at [2j+1, 2i+1] and its four corners at [2j, 2i],
        [2j, 2i+2], [2j+2, 2i+2], [2j+2, 2i].

        The two hemispheres are built by different routes, because they are
        different problems:

        - South: the J-curves are concentric circles, so they are true latitude
          circles and the orthogonal trajectories are true meridians. Rows are
          placed directly at the latitudes in `latitude_bounds_in_SH` and every
          point follows in closed form, radius = tan(pi/4 - lat/2). No ODE.
        - North: the circles are displaced, so each I-curve is obtained by
          integrating I_curve_ray_tracing_system up from the equator with RK4,
          sampling every required v in a single pass.

        The equator is shared: it is the last southern interface and the first
        northern one, and both routes put it on the unit circle, so the two
        halves join without a seam in the mesh lines.

        Parameters
        ----------
        latitude_bounds_in_SH : ascending latitudes [rad] of the southern row
                                edges, ending at 0. Defaults to the value given
                                to __init__.
        v_interfaces_in_NH    : northern row edges in v [rad], ascending from 0.
                                Defaults to linspace(0, pi/2, nj_NH+1) with the
                                topmost edge dropped: there f = g, the J-curve
                                degenerates to a point, and the ray-tracing ODE
                                is 0/0.
        split                 : RK4 substeps between consecutive v targets.
        earth_radius          : used only for e1 and e2, in metres.

        Returns
        -------
        StructuredQuadMesh
        """
        if latitude_bounds_in_SH is None:
            latitude_bounds_in_SH = self.latitude_bounds_in_SH
        latitude_bounds_in_SH = np.asarray(latitude_bounds_in_SH, dtype=float)

        if v_interfaces_in_NH is None:
            v_interfaces_in_NH = np.linspace(0.0, V_MAX, self.number_of_rows_in_NH + 1)[:-1]
        v_interfaces_in_NH = np.asarray(v_interfaces_in_NH, dtype=float)

        ni = self.number_of_columns
        nj_S = latitude_bounds_in_SH.size - 1
        nj_N = v_interfaces_in_NH.size - 1
        nj = nj_S + nj_N

        # doubled row coordinates, interface / centre / interface / ...
        lat_all = np.empty(2*nj_S + 1)
        lat_all[0::2] = latitude_bounds_in_SH
        lat_all[1::2] = 0.5*(latitude_bounds_in_SH[:-1] + latitude_bounds_in_SH[1:])

        v_all = np.empty(2*nj_N + 1)
        v_all[0::2] = v_interfaces_in_NH
        v_all[1::2] = 0.5*(v_interfaces_in_NH[:-1] + v_interfaces_in_NH[1:])

        print("v_all = ", v_all)

        # doubled column coordinates, periodic
        du = 2.0*np.pi/ni
        u_all = np.arange(2*ni) * (du/2.0)

        # southern radii in closed form; the shared equator row is dropped from
        # the northern block so it is not written twice
        radius_S = np.tan(np.pi/4 - lat_all/2.0)
        # integrate_rk4 marches in uniform steps, so the northern row edges must
        # be evenly spaced in v. The doubled array then advances by half a row.
        dv_doubled = np.diff(v_all)
        if not np.allclose(dv_doubled, dv_doubled[0]):
            raise ValueError("v_interfaces_in_NH must be uniformly spaced in v")
        dv_doubled = dv_doubled[0]
        print(f"dv_doubled = {dv_doubled}")
        n_steps_N = v_all.size - 1

        I_curve_generator = functools.partial(I_curve_ray_tracing_system, FG_funcs=self.fg)

        n_rows_doubled = (2*nj_S + 1) + (2*nj_N)
        x = np.zeros((n_rows_doubled, 2*ni))
        y = np.zeros_like(x)
        split_at = 2*nj_S + 1
        for b, u in enumerate(u_all):
            x[:split_at, b] = radius_S*np.cos(u)
            y[:split_at, b] = radius_S*np.sin(u)
            if n_steps_N:
                # out[:, 0] is the equator, already covered by the southern
                # block above, so it is dropped here rather than written twice.
                _, out = integrate_rk4(I_curve_generator, 0.0, [np.cos(u), np.sin(u)],
                                       dv_doubled, n_steps_N, substeps=split)
                x[split_at:, b] = out[0, 1:]
                y[split_at:, b] = out[1, 1:]

        lon, lat = self._stereo_to_spherical(x, y)

        rows_edge   = np.arange(nj) * 2
        rows_edge_1 = rows_edge + 2
        rows_ctr    = rows_edge + 1
        cols_edge   = np.arange(ni) * 2
        cols_edge_1 = (cols_edge + 2) % (2*ni)
        cols_ctr    = cols_edge + 1

        center_lon = lon[np.ix_(rows_ctr, cols_ctr)]
        center_lat = lat[np.ix_(rows_ctr, cols_ctr)]

        corner_lon = np.zeros((nj, ni, 4))
        corner_lat = np.zeros((nj, ni, 4))
        for k, (rr, cc) in enumerate([(rows_edge,   cols_edge),
                                      (rows_edge,   cols_edge_1),
                                      (rows_edge_1, cols_edge_1),
                                      (rows_edge_1, cols_edge)]):
            corner_lon[:, :, k] = lon[np.ix_(rr, cc)]
            corner_lat[:, :, k] = lat[np.ix_(rr, cc)]

        area_sr = _spherical_excess(np.moveaxis(corner_lon, 2, 0),
                                    np.moveaxis(corner_lat, 2, 0))

        return StructuredQuadMesh.from_corners(
            corner_lon = corner_lon,
            corner_lat = corner_lat,
            face_lon   = center_lon,
            face_lat   = center_lat,
            area       = area_sr * earth_radius**2,
            mask       = np.ones((nj, ni), dtype=np.int32),
            shape      = (nj, ni),
            attrs      = {
                "title":                "Displaced pole grid (Madec and Imbard, 1996)",
                "number_of_rows_in_SH": nj_S,
            },
        )


# ---------------------------------------------------------------------------
# Mesh assembly and file output
# ---------------------------------------------------------------------------

def _unit_vectors(lon, lat):
    """Cartesian unit vectors on the sphere from radian lon/lat."""
    return np.stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ), axis=0)



def _spherical_excess(lon, lat):
    """
    Exact solid angle of spherical quadrilaterals bounded by great-circle arcs.

    The cell is split into two triangles and each is evaluated with the
    l'Huilier-free formula of Van Oosterom and Strackee (1983), which stays
    accurate for the very thin slivers that appear next to the mesh pole.

    Parameters
    ----------
    lon, lat : (4, ...) arrays, corners ordered around the cell.

    Returns
    -------
    Solid angle in steradians, shape (...).

    Note this is NOT renormalised to 4 pi: a displaced pole mesh that stops
    short of the pole does not cover the whole sphere, so the unnormalised
    value is the correct one. It is also what ESMF_RegridWeightGen assumes,
    since it treats cell edges as great-circle arcs.
    """
    v = _unit_vectors(lon, lat)

    def triangle(a, b, c):
        numerator = np.abs(np.sum(a * np.cross(b, c, axis=0), axis=0))
        denominator = (1.0
                       + np.sum(a * b, axis=0)
                       + np.sum(b * c, axis=0)
                       + np.sum(c * a, axis=0))
        return 2.0 * np.arctan2(numerator, denominator)

    return (triangle(v[:, 0], v[:, 1], v[:, 2])
            + triangle(v[:, 0], v[:, 2], v[:, 3]))



def build_example_grid(pole_longitude_degree: float = None,
                       number_of_rows_in_NH: int = 30,
                       number_of_columns: int = 120,
                       dlat_in_SH_degree: float = 3.0,
                       southern_edge_degree: float = -90.0,
                       formulation: str = "mi1996"):
    """
    A worked example with a plain lat-lon southern patch and tropical refinement.

    Each formulation carries its own default mesh pole, since the pole latitude
    belongs to the formulation while the longitude belongs to the grid, and the
    two only make sense together -- a latitude that sits on land at one longitude
    sits in open ocean at another. Pass pole_longitude_degree to override.

        mi1996 / polystep    40 N,  90 E   (western China)
        logcosh              72 N,  40 W   (Greenland)

    and a plain lat-lon southern patch.

    dlat_in_SH_degree is also used as the equatorial meridional resolution, so
    that e2 is continuous across the equator. That is the one condition the
    southern patch has to respect; see the note in generate_mesh.

    Note the unit conversion. An FG formulation takes resolutions per unit v,
    not per grid row, so that it stays independent of how finely the grid is
    discretised. Rows enter only through dv/dj, which belongs to the grid:

        dlat/dv = (dlat/dj) / (dv/dj),    dv/dj = (pi/2) / number_of_rows_in_NH

    `formulation` selects which scheme builds f and g; both produce the same
    kind of object and DisplacedPoleGrid cannot tell them apart.
    """
    d2r = np.deg2rad
    dvdj = V_MAX / number_of_rows_in_NH

    if formulation == "mi1996":
        default_pole_lon = 90.0
        fg = FGMI1996(
            displaced_north_pole_lat = d2r(40.0),   # mesh north pole latitude
            dlat_dv_equator          = d2r(dlat_in_SH_degree) / dvdj,
            dlat_dv_polar            = d2r(0.01) / dvdj,
            v_polar                  = d2r(-85.0),
            v_trans_tropics_f        = d2r(20.0),   # tropical refinement knee, f
            v_trans_width_tropics_f  = d2r(10.0),
            v_trans_polar_f          = d2r(70.0),   # polar knee, f
            v_trans_width_polar_f    = d2r(10.0),
            v_trans_g                = d2r(30.0),   # polar knee, g
            v_trans_g_width          = d2r(10.0),
        )
    elif formulation == "linear":
        default_pole_lon = 90.0
        # same binding constraint as polystep, s0 < 2(1 - y_NP)/v_max, so the
        # same 1.5 deg equatorial resolution is needed at 30 northern rows.
        fg = FGLinear(
            displaced_north_pole_lat = d2r(40.0),
            dlat_dv_equator          = d2r(dlat_in_SH_degree) / dvdj,
        )
    elif formulation == "polystep":
        default_pole_lon = 90.0
        # gamma keeps the rate trapped between its endpoints only while the
        # g branch stays monotone, which needs s0 below roughly
        # 2(1 - y_NP)/v_max -- i.e. the equator must be finer than uniform.
        # dlat_in_SH_degree = 1.5 with 30 northern rows gives s0 = 0.5.
        fg = FGPolynomialStep(
            displaced_north_pole_lat = d2r(40.0),
            dlat_dv_equator          = d2r(dlat_in_SH_degree) / dvdj,
            v_width_f                = V_MAX,
        )
    elif formulation == "logcosh":
        # Mesh pole on the Greenland ice sheet, 72 N / 40 W. The construction
        # always builds it at 90 E, so the grid rotates the whole mesh about the
        # Earth's axis to get it there -- see DisplacedPoleGrid.pole_longitude.
        #
        # FGLogCosh takes all four rates as inputs, so feasibility is a condition on
        # the whole set, not on s0 alone. Each rate is trapped between its
        # endpoints, so the mean the geometry demands must lie between them:
        #
        #     min(s0, s_f) <= mean_f <= max(s0, s_f),   mean_f  = 0.7374
        #     s0 >= |mean_g|  and  |s_g| <= |mean_g|,   |mean_g| = 0.5358
        #
        # satisfied here by s0 = 1.0000, s_f = 0.5125, s_g = -0.2858.
        #
        # dg/dv varies by exactly s0 - |s_g| = 0.714, about 130 % of its mean.
        # Flattening it means pushing both onto |mean_g|, i.e. dropping the
        # equator to 1.607 deg/row, which steepens df/dv in exchange -- C1 ties
        # the two starting rates together, so one branch can be flat but not both.
        default_pole_lon = -40.0
        fg = FGLogCosh(
            displaced_north_pole_lat = d2r(72.0),
            dlat_dv_equator          = d2r(dlat_in_SH_degree) / dvdj,
            dlat_dv_polar_f          = 1.000000,
            dlat_dv_polar_g          = 0.557594,
            v_width_f                = d2r(30.0),
            v_width_g                = d2r(30.0),
        )
    elif formulation == "cubic":
        default_pole_lon = 90.0
        fg = FGCubic(
            displaced_north_pole_lat = d2r(40.0),
            dlat_dv_equator          = d2r(dlat_in_SH_degree) / dvdj,
            dlat_dv_polar_f          = d2r(dlat_in_SH_degree) / dvdj,
            dlat_dv_polar_g          = d2r(dlat_in_SH_degree) / dvdj,
        )
    else:
        raise ValueError(f"unknown formulation {formulation!r}")

    if pole_longitude_degree is None:
        pole_longitude_degree = default_pole_lon

    return DisplacedPoleGrid(
        fg                       = fg,
        pole_longitude           = d2r(pole_longitude_degree),
        number_of_rows_in_NH     = number_of_rows_in_NH,
        # stop short of -90: there all meridians converge and e1 -> 0, the
        # ordinary lat-lon pole singularity. Antarctica covers the remainder.
        latitude_bounds_in_SH    = d2r(np.arange(southern_edge_degree, 1e-9,
                                                 dlat_in_SH_degree)),
        number_of_columns        = number_of_columns,
    )


def test_output_grid_file(scrip_file: str = "grid_displaced_pole_SCRIP.nc",
                           cf_file: str = "grid_displaced_pole_cf.nc",
                           formulation: str = "mi1996",
                           mask_land: bool = True, **grid_kwargs):
    """
    Write both output files: the SCRIP grid for ESMF_RegridWeightGen, and a
    CF-1.10 / UGRID-1.0 grid file via write_to_CF_grid_file.
    """
    print("Generating grid...")
    grid = build_example_grid(formulation=formulation, **grid_kwargs)
    print(f"  formulation: {type(grid.fg).__name__}")

    print("Assembling mesh...")
    mesh = grid.generate_mesh()
    nj, ni = mesh.shape
    nj_S = mesh.attrs.get("number_of_rows_in_SH", "?")
    print(f"  {nj} x {ni} cells, {nj_S} of them south of the equator")
    print(f"  latitude {np.rad2deg(mesh.face_lat.min()):+.2f} .. "
          f"{np.rad2deg(mesh.face_lat.max()):+.2f} deg")
    print(f"  total area {mesh.area.sum()/1e12:.3f} million km²")

    if mask_land:
        apply_ocean_mask(mesh)
        ocean = mesh.mask.sum()
        frac  = (mesh.area * mesh.mask).sum() / mesh.area.sum()
        print(f"  ocean mask: {ocean} of {mesh.mask.size} cells wet "
              f"({100*ocean/mesh.mask.size:.1f}% by count, "
              f"{100*frac:.1f}% by area)")

    print("Writing to file: ", scrip_file)
    mesh.write_to_SCRIP_grid_file(scrip_file)

    print("Writing to file: ", cf_file)
    mesh.write_to_CF_grid_file(cf_file)


def test_plot_grid_naive(output_file: str = None,
                             formulation: str = "mi1996",
                             v_min_degree: float = 0.0,
                             **grid_kwargs):
 
    import matplotlib.pyplot as plt

    grid = build_example_grid(formulation=formulation, **grid_kwargs)

    start_lons = np.deg2rad(np.linspace(0, 360, 10)[:-1])
    I_curves = [ grid.generate_I_curve(lon) for lon in start_lons ]
    
    start_lats = np.deg2rad(np.linspace(-90, 90, 20))
    J_curves = [ grid.generate_J_curve(lat) for lat in start_lats ]

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.view_init(azim=-30, elev=45, roll=0)

    for curves in [ I_curves, J_curves ]:
        for i, pts in enumerate(curves):
            xyz = spherical_to_cartesian(pts)
            x = xyz[0, :]
            y = xyz[1, :]
            z = xyz[2, :]
            ax.plot(x, y, z, lw=0.9)

    ax.set_xlabel("x-direction")
    ax.set_ylabel("y-direction")
    ax.set_zlabel("z-direction")
    lim = np.array([-1, 1]) * 1.1
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    plt.show()

def test_plot_grid(stride_j: int = 1, stride_i: int = 1, output_file: str = None,
                   formulation: str = "mi1996", **grid_kwargs):
    """
    Draw the mesh lines on the sphere. Every stride_j-th row and stride_i-th
    column of the assembled mesh is shown, so what is plotted is exactly what
    gets written to file.
    """
    import matplotlib.pyplot as plt

    grid = build_example_grid(formulation=formulation, **grid_kwargs)
    mesh = grid.generate_mesh()
    nj, ni = mesh.shape
    lon = mesh.node_lon[mesh.face_nodes[:, 0]].reshape(nj, ni)
    lat = mesh.node_lat[mesh.face_nodes[:, 0]].reshape(nj, ni)
    xyz = spherical_to_cartesian(np.stack((lon.ravel(), lat.ravel()), axis=0))
    x, y, z = (c.reshape(lon.shape) for c in xyz)
    print("lon.shape = ", lon.shape)
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.view_init(azim=-30, elev=45, roll=0)
    for j in range(0, lon.shape[0], stride_j):              # J-curves
        ax.plot(np.append(x[j], x[j, 0]), np.append(y[j], y[j, 0]),
                np.append(z[j], z[j, 0]), lw=0.6, color="0.45")
    for i in range(0, lon.shape[1], stride_i):              # I-curves
        ax.plot(x[:, i], y[:, i], z[:, i], lw=0.9)

    ax.set_xlabel("x-direction")
    ax.set_ylabel("y-direction")
    ax.set_zlabel("z-direction")
    lim = np.array([-1, 1]) * 1.1
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print("Wrote plot: ", output_file)


def test_plot_fg_derivatives(output_file: str = None,
                             formulation: str = "mi1996",
                             v_min_degree: float = 0.0,
                             **grid_kwargs):
    """
    Three panels for a formulation, reading top-down as value, rate, and
    physical consequence:

    - top: f and g themselves. Monotonicity is directly visible here rather than
      inferred, and the vertical gap between the curves is twice the J-curve
      radius, so property (a) -- strictly embedded circles -- is the statement
      that the two curves never touch before the mesh pole.
    - middle: df/dv and dg/dv, signed. These are what the boundary conditions
      pin, so they are readable straight off the axes. Zero crossings are marked
      because that is exactly where a branch reverses.
    - bottom: the same information as meridional grid spacing in degrees per row,
      dlat/dj = |d/dv| * 2/(1 + w^2) * dv/dj. The Jacobian is far from 1 away
      from the equator, so this does not have the shape of the middle panel, and
      it is the one to judge resolution by.

    Note the bottom panel is a MAGNITUDE. A branch reversing shows up there as a
    touch-down to zero, which reads like very fine rows rather than a failure --
    which is why the middle panel marks the crossings explicitly.

    v < 0 is shaded when plotted: a formulation may still be defined there, and
    a boundary condition may still act there, but the mesh does not use it --
    the southern hemisphere is the lat-lon patch.

    Formulation-agnostic: it reads only grid.fg's four methods, s0 and
    transition_marks(), so it works for any FGBase subclass.
    """
    import matplotlib.pyplot as plt

    grid = build_example_grid(formulation=formulation, **grid_kwargs)
    fg = grid.fg
    r2d = np.rad2deg

    v = np.linspace(np.deg2rad(v_min_degree), V_MAX, 1201)
    dfdv = np.array([fg.dfdv(vv) for vv in v])
    dgdv = np.array([fg.dgdv(vv) for vv in v])
    f_v  = np.array([fg.f(vv) for vv in v])
    g_v  = np.array([fg.g(vv) for vv in v])

    def zero_crossings(y):
        idx = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
        return [v[i] - y[i]*(v[i+1]-v[i])/(y[i+1]-y[i]) for i in idx]

    crossings = [(vc, "f'") for vc in zero_crossings(dfdv)] + \
                [(vc, "g'") for vc in zero_crossings(dgdv)]

    # ordinate rate -> meridional spacing in degrees per row
    to_deg_per_row = lambda d, w: r2d(np.abs(d) * 2.0/(1.0 + w**2) * grid.dvdj)

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    ax = axes[0]
    ax.plot(r2d(v), f_v, label="f  (crossing on lon 270)", lw=1.6)
    ax.plot(r2d(v), g_v, label="g  (crossing on lon 90)", lw=1.6)
    ax.fill_between(r2d(v), f_v, g_v, color="0.85", zorder=0,
                    label="J-curve diameter (g - f)")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.plot([0, 0, r2d(V_MAX)], [-1.0, 1.0, fg.y_NP], "k.", ms=8, zorder=5)
    ax.annotate(f"  f(0) = -1", (0, -1.0), fontsize=8, va="top")
    ax.annotate(f"  g(0) = +1", (0, 1.0), fontsize=8, va="bottom")
    ax.annotate(f"f = g = y_NP = {fg.y_NP:.4f}  ", (r2d(V_MAX), fg.y_NP),
                fontsize=8, va="bottom", ha="right")
    ax.set_ylabel("stereographic ordinate")
    ax.set_title(f"{type(fg).__name__}: f and g "
                 f"(monotonicity and embedding are visible here)")
    ax.legend(loc="center left", fontsize=9)

    ax = axes[1]
    ax.plot(r2d(v), dfdv, label="df/dv", lw=1.6)
    ax.plot(r2d(v), dgdv, label="dg/dv", lw=1.6)
    ax.axhline(0.0, color="0.7", lw=0.8)
    s0 = fg.s0
    ax.plot([0, 0], [s0, -s0], "k.", ms=8, zorder=5)
    ax.annotate(f"  f'(0) = +s0 = {s0:.4f}", (0, s0), fontsize=8, va="bottom")
    ax.annotate(f"  g'(0) = -s0", (0, -s0), fontsize=8, va="top")
    for vc, which in crossings:
        ax.plot(r2d(vc), 0.0, "rx", ms=9, mew=2, zorder=6)
        ax.annotate(f" {which} = 0 at {r2d(vc):.1f} deg", (r2d(vc), 0.0),
                    fontsize=8, color="red", va="bottom")
    ax.set_ylabel("d/dv  [stereographic radius per rad of v]")
    ax.set_title("Shape derivatives (what the boundary conditions pin)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[2]
    ax.plot(r2d(v), to_deg_per_row(dfdv, f_v), label="f branch (lon 270 / 90)", lw=1.6)
    ax.plot(r2d(v), to_deg_per_row(dgdv, g_v), label="g branch (lon 90)", lw=1.6)
    ax.axhline(r2d(fg.s0 * grid.dvdj), color="0.5", ls=":", lw=1.0,
               label="equatorial resolution")
    ax.set_ylabel("meridional spacing  [deg / row]   (magnitude)")
    ax.set_xlabel("v  [deg]")
    ax.set_title("Same thing as grid resolution (paper Fig. 2)")
    ax.legend(loc="upper left", fontsize=9)

    for ax in axes:
        if v[0] < 0.0:
            ax.axvspan(r2d(v[0]), 0.0, color="0.9", zorder=0)
        for vt, name in fg.transition_marks():
            if v[0] <= vt <= v[-1] and abs(vt - V_MAX) > 1e-9:
                ax.axvline(r2d(vt), color="0.3", ls="--", lw=0.7, alpha=0.5)
        for vc, _ in crossings:
            ax.axvline(r2d(vc), color="red", ls=":", lw=1.0, alpha=0.7)
        ax.axvline(90.0, color="k", lw=0.8, alpha=0.6)
        ax.grid(alpha=0.25)

    if v[0] < 0.0:
        axes[0].annotate("not used by the mesh\n(lat-lon patch)",
                         (r2d(v[0]) * 0.95, axes[0].get_ylim()[1]),
                         fontsize=7, color="0.4", va="top")
    for vt, name in fg.transition_marks():
        if v[0] <= vt <= v[-1] and abs(vt - V_MAX) > 1e-9:
            axes[1].annotate(name, (r2d(vt), axes[1].get_ylim()[0]), fontsize=6,
                             rotation=90, va="bottom", ha="right", color="0.3")
    axes[1].annotate("mesh pole", (90.0, axes[1].get_ylim()[0]), fontsize=7,
                     rotation=90, va="bottom", ha="right")

    fig.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print("Wrote plot: ", output_file)


def test_plot_stereographic(output_file: str = None,
                            formulation: str = "logcosh",
                            stride_j: int = 2,
                            stride_i: int = 4,
                            max_radius: float = 2.0,
                            coastline_resolution: str = "110m",
                            **grid_kwargs):
    """
    Draw the mesh in the north polar stereographic plane it is constructed in,
    with a coastline on top.

    This is the view the construction is actually designed in, and it shows what
    the lon/lat and 3-D plots cannot:

    - the J-curves are literally circles here, so "strictly embedded" is visible
      as circles nesting without touching;
    - the I-curves meet them at right angles on the page, because the projection
      is conformal -- so orthogonality can be checked by eye;
    - the origin is the GEOGRAPHIC north pole and the unit circle is the equator,
      so the offset between the origin and the point where the mesh meridians
      converge IS the pole displacement, drawn to scale.

    max_radius bounds the view: radius = tan(pi/4 - lat/2), so 1 is the equator
    and 2 is about 37 S. The southern rows run to radius ~76 at 88.5 S, which is
    why the plot has to be clipped rather than showing the whole mesh.

    Coastlines come from the Natural Earth data cartopy caches locally.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import cartopy.io.shapereader as shpreader

    grid = build_example_grid(formulation=formulation, **grid_kwargs)
    mesh = grid.generate_mesh()
    nj, ni = mesh.shape
    lon = mesh.node_lon[mesh.face_nodes[:, 0]].reshape(nj, ni)
    lat = mesh.node_lat[mesh.face_nodes[:, 0]].reshape(nj, ni)
    x, y = spherical_to_stereo(lon, lat)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # coastline, projected into the same plane and split where it leaves the view
    path = shpreader.natural_earth(resolution=coastline_resolution,
                                   category="physical", name="coastline")
    for geom in shpreader.Reader(path).geometries():
        for line in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            c = np.asarray(line.coords)
            cx, cy = spherical_to_stereo(np.deg2rad(c[:, 0]), np.deg2rad(c[:, 1]))
            inside = np.hypot(cx, cy) <= max_radius
            if not inside.any():
                continue
            # break the polyline wherever it leaves the clip, so no false chords
            breaks = np.where(np.diff(inside.astype(int)) != 0)[0] + 1
            for seg_x, seg_y, seg_in in zip(np.split(cx, breaks),
                                            np.split(cy, breaks),
                                            np.split(inside, breaks)):
                if seg_in.all() and seg_x.size > 1:
                    ax.plot(seg_x, seg_y, color="0.35", lw=0.7, zorder=3)

    for j in range(0, lat.shape[0], stride_j):                 # J-curves
        ax.plot(np.append(x[j], x[j, 0]), np.append(y[j], y[j, 0]),
                color="C0", lw=0.5, alpha=0.65, zorder=2)
    for i in range(0, lat.shape[1], stride_i):                 # I-curves
        ax.plot(x[:, i], y[:, i], color="C1", lw=0.5, alpha=0.65, zorder=2)

    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="k", lw=1.2,
                        ls="--", zorder=4))
    ax.plot(0, 0, "k+", ms=12, mew=1.6, zorder=5)
    # The construction puts the pole on the +y axis (90 E); pole_longitude
    # rotates the whole mesh about the Earth's axis, so the marker follows.
    xp, yp = spherical_to_stereo(grid.pole_longitude,
                                 grid.fg.displaced_north_pole_lat)
    ax.plot(xp, yp, "r*", ms=15, zorder=6)
    ax.annotate("geographic N pole", (0, 0), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.annotate(f"mesh pole "
                f"({np.rad2deg(grid.fg.displaced_north_pole_lat):.0f}N, "
                f"{np.rad2deg(grid.pole_longitude):.0f}E)",
                (xp, yp), textcoords="offset points", xytext=(10, -14),
                fontsize=8, color="red")
    ax.annotate("equator", (0, -1.0), textcoords="offset points",
                xytext=(4, 4), fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_xlabel("stereographic x"); ax.set_ylabel("stereographic y")
    ax.set_title(f"{type(grid.fg).__name__}: mesh in the stereographic plane "
                 f"(J blue, I orange)")
    ax.grid(alpha=0.2)

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print("Wrote plot: ", output_file)


if __name__ == "__main__":

    # Each formulation needs its own equatorial resolution. The locked schemes
    # (linear, polystep) cap s0 at 2(1 - g_target)/v_max, which is 0.68 for a
    # 40 N pole, so they need a finer equator than the 3 deg default.
    FORMULATIONS = [
        ("logcosh",     dict()),
        #("mi1996",  dict(v_min_degree=-90.0)),
        #("cubic",    dict()),
        #("polystep", dict(dlat_in_SH_degree=1.5)),
        #("linear",   dict(dlat_in_SH_degree=1.5)),
    ]

    for name, kwargs in FORMULATIONS:
        print(f"--- {name} ---")

        # v_min_degree belongs to the derivative plot only
        mesh_kwargs = {k: v for k, v in kwargs.items() if k != "v_min_degree"}

        """
        print("  f' and g' ...")
        test_plot_fg_derivatives(output_file=f"figure_fg_derivative_{name}.svg",
                                 formulation=name, **kwargs)

        print("  mesh on the sphere ...")
        test_plot_grid(output_file=f"figure_grid_{name}.svg",
                       formulation=name, stride_j=4, stride_i=6, **mesh_kwargs)

        print("  mesh in the stereographic plane ...")
        test_plot_stereographic(output_file=f"figure_stereographic_{name}.svg",
                                formulation=name, **mesh_kwargs)
        """
        print("  grid files ...")
        test_output_grid_file(scrip_file=f"grid_displaced_pole_{name}_SCRIP.nc",
                               cf_file=f"grid_displaced_pole_{name}_CF.nc",
                               formulation=name, **mesh_kwargs)
