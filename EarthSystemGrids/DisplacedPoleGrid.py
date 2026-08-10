import abc
import functools

import numpy as np

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

    Subclasses must implement f, dfdv, g, dgdv, and may override
    `_scheme_checks` and `transition_marks`.
    """

    def __init__(self, displaced_north_pole_lat, dlat_dv_equator, v_max=np.pi/2):
        self.displaced_north_pole_lat = displaced_north_pole_lat
        self.v_max = v_max
        _, self.y_NP = spherical_to_stereo(np.pi/2, displaced_north_pole_lat)
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
            ("f(v_max) = y_NP",             self.f(self.v_max),               self.y_NP, "eq"),
            ("g(v_max) = y_NP",             self.g(self.v_max),               self.y_NP, "eq"),
            ("C1 at equator: f'+g' = 0",    self.dfdv(0.0) + self.dgdv(0.0),   0.0,      "eq"),
            ("(c) pole closes: f-g = 0",    f_v[-1] - g_v[-1],                 0.0,      "eq"),
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


class FGLogCosh(FGBase):
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
                 v_max=np.pi/2):
        super().__init__(displaced_north_pole_lat, dlat_dv_equator, v_max)
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
            np.array([s0, 1 - self.y_NP]),
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
            np.array([s0, s_star, 1 + self.y_NP]),
        )

    def _scheme_checks(self):
        return [
            ("dfdv(v_polar) = s_polar", self.dfdv(self._v_star), self._s_star, "eq"),
            ("design: W_polar_f   > 0", self.W_polar_f,          0.0,          "gt"),
            ("design: W_tropics_f > 0", self.W_tropics_f,        0.0,          "gt"),
            ("design: W_g         > 0", self.W_g,                0.0,          "gt"),
        ]


class FGCubic(FGBase):
    """
    Cubic Hermite formulation: f and g are the unique cubics matching a value
    and a slope at each end of [0, v_max].

        f(v) = c0 + c1 v + c2 v^2 + c3 v^3

    Four coefficients, four conditions, and the interpolant is available in
    closed form, so this scheme has no solver at all -- `_solve` is a few lines
    of arithmetic rather than two linear systems. There are also no transition
    parameters to choose.

    Boundary conditions
    -------------------
        f(0) = -1,  dfdv(0) = +s0        g(0) = +1,  dgdv(0) = -s0
        f(v_max) = y_NP,  dfdv(v_max)    g(v_max) = y_NP,  dgdv(v_max)

    Two properties this buys over a scheme with mismatched bases:

    - g gets a fourth condition, so its polar slope is specifiable. In the
      log-cosh scheme g has only three parameters and dgdv(v_max) is whatever
      falls out.
    - because f and g are drawn from the SAME basis, mirrored boundary data
      give exactly mirrored polynomials. Setting the mesh pole to 90 N with
      mirrored polar slopes therefore yields g = -f identically, i.e. concentric
      circles and a plain lat-lon grid, to the last bit.

    What it gives up is local control. The log-cosh ramps let the paper place a
    knee at a chosen latitude; a cubic is pinned only at its ends, and whatever
    happens in between is forced by the endpoint data. Note that the mid-range
    excursion is largely geometry rather than basis: f must cover the whole
    continued-latitude span between its endpoints, so if both ends are held
    below the mean rate the middle has to exceed it.
    """

    def __init__(self,
                 displaced_north_pole_lat,      # [rad]
                 dlat_dv_equator,               # [rad lat / rad v]
                 dlat_dv_polar_f,               # [rad lat / rad v] at v_max, f branch
                 dlat_dv_polar_g,               # [rad lat / rad v] at v_max, g branch
                 v_max=np.pi/2):
        super().__init__(displaced_north_pole_lat, dlat_dv_equator, v_max)
        self.dlat_dv_polar_f = dlat_dv_polar_f
        self.dlat_dv_polar_g = dlat_dv_polar_g
        self._solve()
        if not self.verify(verbose=False):
            print(f"{type(self).__name__}: verify() reported failures; "
                  f"call verify() for the detail.")

    @staticmethod
    def _hermite(y0, m0, y1, m1, L):
        """Cubic Hermite on [0, L] in monomial form, c0 + c1 v + c2 v^2 + c3 v^3."""
        D = (y1 - y0) / L
        return np.array([y0, m0, (3*D - 2*m0 - m1)/L, (m0 + m1 - 2*D)/L**2])

    def _solve(self):
        # The f endpoint sits at y_NP > 0, i.e. PAST the zero crossing where the
        # -y crossing has gone over the geographic pole. The folded-latitude
        # Jacobian flips sign there, so use the continued-latitude form.
        m1_f = self.dlat_dv_to_dfdv(self.dlat_dv_polar_f, self.y_NP)
        m1_g = self.dlat_dv_to_dgdv(self.dlat_dv_polar_g, self.y_NP)

        self.cf = self._hermite(-1.0,  self.s0, self.y_NP, m1_f, self.v_max)
        self.cg = self._hermite(+1.0, -self.s0, self.y_NP, m1_g, self.v_max)

    @staticmethod
    def _poly(c, v):
        return c[0] + c[1]*v + c[2]*v**2 + c[3]*v**3

    @staticmethod
    def _dpoly(c, v):
        return c[1] + 2*c[2]*v + 3*c[3]*v**2

    def f(self, v):
        return self._poly(self.cf, v)

    def dfdv(self, v):
        return self._dpoly(self.cf, v)

    def g(self, v):
        if v < 0:
            return - self.f(v)
        return self._poly(self.cg, v)

    def dgdv(self, v):
        if v < 0:
            return - self.dfdv(v)
        return self._dpoly(self.cg, v)

    @staticmethod
    def _monotone_margin(alpha, beta):
        """
        Exact monotonicity test for a cubic Hermite (Fritsch and Carlson, 1980),
        in terms of the endpoint slopes scaled by the secant, alpha = m0/D and
        beta = m1/D. Returns a value that is positive exactly when the
        interpolant is monotone.

        The familiar alpha^2 + beta^2 <= 9 disk is only a SUFFICIENT condition --
        an inscribed approximation of the true region -- so testing against it
        reports failures for interpolants that are perfectly monotone. The full
        region is the union of three half-planes and one curved piece.
        """
        if alpha < 0.0 or beta < 0.0:
            return min(alpha, beta)
        if (alpha + beta - 2.0 <= 0.0
                or alpha + 2.0*beta - 3.0 <= 0.0
                or 2.0*alpha + beta - 3.0 <= 0.0):
            return 1.0                      # inside the half-plane part; any +ve means monotone
        return alpha - (2.0*alpha + beta - 3.0)**2 / (3.0*(alpha + beta - 2.0))

    def _scheme_checks(self):
        """
        Monotonicity established a priori from the endpoint data, rather than by
        sampling as the base class does. For a cubic this is exact and costs
        nothing, which is one of the quieter advantages of a closed-form scheme.
        """
        rows = []
        for name, c, y0, y1 in [("f", self.cf, -1.0, self.y_NP),
                                ("g", self.cg, +1.0, self.y_NP)]:
            D = (y1 - y0) / self.v_max
            a = self._dpoly(c, 0.0) / D
            b = self._dpoly(c, self.v_max) / D
            rows.append((f"{name}: monotone a priori", self._monotone_margin(a, b), 0.0, "gt"))
        return rows


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
    ):
        self.fg = fg
        self.number_of_rows_in_NH = number_of_rows_in_NH
        self.latitude_bounds_in_SH = np.asarray(latitude_bounds_in_SH, dtype=float)
        self.number_of_columns = number_of_columns

        # v parameterises only the northern branch, equator to mesh pole. dv/dj
        # carries the resolution and `fg` never sees it, which is what keeps the
        # formulation independent of how finely the grid is discretised.
        self.dvdj = (np.pi/2) / number_of_rows_in_NH

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

            lon, lat = stereo_to_spherical(x, y)

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

        lon, lat = stereo_to_spherical(pts_xy[0, :], pts_xy[1, :])
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
        DisplacedPoleMesh
        """
        if latitude_bounds_in_SH is None:
            latitude_bounds_in_SH = self.latitude_bounds_in_SH
        latitude_bounds_in_SH = np.asarray(latitude_bounds_in_SH, dtype=float)

        if v_interfaces_in_NH is None:
            v_interfaces_in_NH = np.linspace(0.0, np.pi/2, self.number_of_rows_in_NH + 1)[:-1]
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

        lon, lat = stereo_to_spherical(x, y)

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

        # scale factors straight off the doubled mesh: the mid-edge points are
        # already there, so no interpolation or finite differencing is needed.
        e1 = _great_circle(lon[np.ix_(rows_ctr, cols_edge)],   lat[np.ix_(rows_ctr, cols_edge)],
                           lon[np.ix_(rows_ctr, cols_edge_1)], lat[np.ix_(rows_ctr, cols_edge_1)],
                           earth_radius)
        e2 = _great_circle(lon[np.ix_(rows_edge, cols_ctr)],   lat[np.ix_(rows_edge, cols_ctr)],
                           lon[np.ix_(rows_edge_1, cols_ctr)], lat[np.ix_(rows_edge_1, cols_ctr)],
                           earth_radius)

        area = _spherical_excess(np.moveaxis(corner_lon, 2, 0),
                                 np.moveaxis(corner_lat, 2, 0))

        return DisplacedPoleMesh(
            center_lon = center_lon,
            center_lat = center_lat,
            corner_lon = corner_lon,
            corner_lat = corner_lat,
            e1 = e1,
            e2 = e2,
            area = area,
            mask = np.ones((nj, ni), dtype=np.int32),
            number_of_rows_in_SH = nj_S,
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


def _great_circle(lon1, lat1, lon2, lat2, radius: float = 1.0):
    """Great-circle distance between two points, in the units of `radius`."""
    a = _unit_vectors(lon1, lat1)
    b = _unit_vectors(lon2, lat2)
    return radius * np.arccos(np.clip(np.sum(a * b, axis=0), -1.0, 1.0))


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


class DisplacedPoleMesh:
    """
    A concrete set of cells produced by DisplacedPoleGrid.generate_mesh().

    All longitudes and latitudes are in radians. Corners are ordered
    counter-clockwise starting from the (v_low, u_low) corner, which is the
    ordering ESMF_RegridWeightGen expects.

    Attributes
    ----------
    center_lon, center_lat : (nj, ni)
    corner_lon, corner_lat : (nj, ni, 4)
    e1, e2                 : (nj, ni) zonal and meridional scale factors [m]
    area                   : (nj, ni) solid angle [steradian]
    mask                   : (nj, ni) int, 1 = active
    number_of_rows_in_SH   : rows below the equator (the lat-lon patch)
    """

    def __init__(self, center_lon, center_lat, corner_lon, corner_lat,
                 e1, e2, area, mask, number_of_rows_in_SH):
        self.center_lon = center_lon
        self.center_lat = center_lat
        self.corner_lon = corner_lon
        self.corner_lat = corner_lat
        self.e1 = e1
        self.e2 = e2
        self.area = area
        self.mask = mask
        self.number_of_rows_in_SH = number_of_rows_in_SH

    @property
    def shape(self):
        return self.center_lon.shape


def write_to_SCRIP_grid_file(mesh: DisplacedPoleMesh, output_file, flatten: bool = True):
    """
    Write `mesh` as a SCRIP grid file, readable by ESMF_RegridWeightGen.

    With flatten=True the cells are collapsed onto a single `grid_size`
    dimension and written in degrees, which is the portable form. With
    flatten=False the (lat, lon) structure is kept and radians are used,
    which is easier to inspect but relies on the reader honouring grid_dims.
    """
    import xarray as xr

    nj, ni = mesh.shape
    grid_corners = 4

    # ESMF_RegridWeightGen reads grid_dims in the reverse of the array order.
    # This is undocumented in the user manual; see also JCMGrid.py.
    grid_dims = [ni, nj]
    grid_dim_names = ["lat", "lon"]

    rad2deg = 180.0 / np.pi

    if flatten:
        ds = xr.Dataset(
            data_vars = dict(
                grid_dims       = ( ["grid_rank", ], grid_dims),
                grid_imask      = ( ["grid_size", ], mesh.mask.flatten()),
                grid_center_lat = ( ["grid_size", ], mesh.center_lat.flatten() * rad2deg, {"units" : "degrees"} ),
                grid_center_lon = ( ["grid_size", ], mesh.center_lon.flatten() * rad2deg, {"units" : "degrees"} ),
                grid_corner_lat = ( ["grid_size", "grid_corners"], mesh.corner_lat.reshape((-1, grid_corners)) * rad2deg, {"units" : "degrees"} ),
                grid_corner_lon = ( ["grid_size", "grid_corners"], mesh.corner_lon.reshape((-1, grid_corners)) * rad2deg, {"units" : "degrees"} ),
                grid_area       = ( ["grid_size", ], mesh.area.flatten(), {"units" : "radians^2"} ),
            ),
        )
    else:
        ds = xr.Dataset(
            data_vars = dict(
                grid_dims       = ( ["grid_rank", ], grid_dims),
                grid_imask      = ( [*grid_dim_names], mesh.mask),
                grid_center_lat = ( [*grid_dim_names], mesh.center_lat, {"units" : "radians"} ),
                grid_center_lon = ( [*grid_dim_names], mesh.center_lon, {"units" : "radians"} ),
                grid_corner_lat = ( [*grid_dim_names, "grid_corners"], mesh.corner_lat, {"units" : "radians"} ),
                grid_corner_lon = ( [*grid_dim_names, "grid_corners"], mesh.corner_lon, {"units" : "radians"} ),
                grid_area       = ( [*grid_dim_names], mesh.area, {"units" : "radians^2"} ),
            ),
        )

    ds.attrs["title"] = "Displaced pole grid (Madec and Imbard, 1996)"
    ds.to_netcdf(output_file)


def write_to_2D_grid_file(mesh: DisplacedPoleMesh, output_file):
    """
    Write `mesh` as a plain 2D (j, i) file for quick inspection in ncview.

    ncview cannot read the SCRIP layout, which flattens every cell onto one
    dimension. Here each field keeps its (j, i) shape and carries a
    `coordinates = "lon lat"` attribute, so ncview will offer lon/lat as
    curvilinear axes and every diagnostic can be displayed as an image.
    """
    import xarray as xr

    nj, ni = mesh.shape
    ratio = np.maximum(mesh.e1 / mesh.e2, mesh.e2 / mesh.e1)
    rad2deg = 180.0 / np.pi

    coords = {"j": np.arange(nj), "i": np.arange(ni)}
    field = lambda data, units, long_name: (
        ["j", "i"], data, {"units": units, "long_name": long_name, "coordinates": "lon lat"}
    )

    ds = xr.Dataset(
        data_vars = dict(
            lon        = ( ["j", "i"], mesh.center_lon * rad2deg, {"units": "degrees_east",  "long_name": "cell centre longitude"} ),
            lat        = ( ["j", "i"], mesh.center_lat * rad2deg, {"units": "degrees_north", "long_name": "cell centre latitude"} ),
            e1         = field(mesh.e1 / 1.0e3, "km",         "zonal scale factor"),
            e2         = field(mesh.e2 / 1.0e3, "km",         "meridional scale factor"),
            anisotropy = field(ratio,           "1",          "max(e1/e2, e2/e1)"),
            area       = field(mesh.area,       "steradian",  "cell solid angle"),
            mask       = field(mesh.mask,       "1",          "1 = active cell"),
        ),
        coords = coords,
    )
    ds.attrs["title"] = "Displaced pole grid (Madec and Imbard, 1996), 2D view for ncview"
    ds.to_netcdf(output_file)


def build_example_grid(number_of_rows_in_NH: int = 30,
                       number_of_columns: int = 120,
                       dlat_in_SH_degree: float = 3.0,
                       southern_edge_degree: float = -90.0,
                       formulation: str = "logcosh"):
    """
    A worked example: mesh pole over China at 40 N / 90 E, tropical refinement,
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
    dvdj = (np.pi/2) / number_of_rows_in_NH

    if formulation == "logcosh":
        fg = FGLogCosh(
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
    elif formulation == "cubic":
        fg = FGCubic(
            displaced_north_pole_lat = d2r(40.0),
            dlat_dv_equator          = d2r(dlat_in_SH_degree) / dvdj,
            dlat_dv_polar_f          = d2r(dlat_in_SH_degree) / dvdj,
            dlat_dv_polar_g          = d2r(dlat_in_SH_degree) / dvdj,
        )
    else:
        raise ValueError(f"unknown formulation {formulation!r}")

    return DisplacedPoleGrid(
        fg                       = fg,
        number_of_rows_in_NH     = number_of_rows_in_NH,
        # stop short of -90: there all meridians converge and e1 -> 0, the
        # ordinary lat-lon pole singularity. Antarctica covers the remainder.
        latitude_bounds_in_SH    = d2r(np.arange(southern_edge_degree, 1e-9,
                                                 dlat_in_SH_degree)),
        number_of_columns        = number_of_columns,
    )


def test_output_SCRIP_file(scrip_file: str = "grid_displaced_pole_SCRIP.nc",
                           twod_file: str = "grid_displaced_pole_2D.nc"):
    """
    Write both output files: the SCRIP grid for ESMF_RegridWeightGen, and a
    plain (j, i) file that ncview can display directly.
    """
    print("Generating grid...")
    grid = build_example_grid()
    print(f"  formulation: {type(grid.fg).__name__}")

    print("Assembling mesh...")
    mesh = grid.generate_mesh()
    nj, ni = mesh.shape
    print(f"  {nj} x {ni} cells, {mesh.number_of_rows_in_SH} of them south of the equator")
    print(f"  latitude {np.rad2deg(mesh.center_lat.min()):+.2f} .. "
          f"{np.rad2deg(mesh.center_lat.max()):+.2f} deg")
    print(f"  e1 {mesh.e1.min()/1e3:.1f} .. {mesh.e1.max()/1e3:.1f} km, "
          f"e2 {mesh.e2.min()/1e3:.1f} .. {mesh.e2.max()/1e3:.1f} km")
    print(f"  total solid angle {mesh.area.sum():.6f} sr")

    print("Writing to file: ", scrip_file)
    write_to_SCRIP_grid_file(mesh, scrip_file)

    print("Writing to file: ", twod_file)
    write_to_2D_grid_file(mesh, twod_file)


def test_plot_grid_naive():
    import matplotlib.pyplot as plt

    grid = build_example_grid()

    start_lons = np.deg2rad([0, 90, 180, 270])
    I_curves = [ grid.generate_I_curve(lon) for lon in start_lons ]

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.view_init(azim=-30, elev=45, roll=0)

    for i, pts in enumerate(I_curves):
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

def test_plot_grid(stride_j: int = 1, stride_i: int = 1, output_file: str = None):
    """
    Draw the mesh lines on the sphere. Every stride_j-th row and stride_i-th
    column of the assembled mesh is shown, so what is plotted is exactly what
    gets written to file.
    """
    import matplotlib.pyplot as plt

    grid = build_example_grid()
    mesh = grid.generate_mesh()
    lon = mesh.corner_lon[:, :, 0]
    lat = mesh.corner_lat[:, :, 0]
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
        plt.savefig(output_file, dpi=110, bbox_inches="tight")
        print("Wrote plot: ", output_file)


def test_plot_fg_derivatives(output_file: str = None):
    """
    Plot df/dv and dg/dv against v, the analogue of Fig. 2 of Madec and Imbard
    (1996), where the envelope of the e2 curves is exactly these derivatives.

    Two panels, because the two quantities answer different questions:

    - top: the raw shape derivatives. These are what the formulation's boundary
      conditions pin, so they are directly readable off the axes.
    - bottom: the same information converted to meridional grid spacing in
      degrees per row, via dlat/dj = |d/dv| * 2/(1 + w^2) * dv/dj. That factor
      is not close to 1 away from the equator, so the two panels do not have
      the same shape, and the bottom one is the one to judge resolution by.

    v < 0 is shaded: a formulation may still be defined there, and a boundary
    condition may still act there, but the mesh does not use it -- the southern
    hemisphere is the lat-lon patch.

    Formulation-agnostic: it reads only grid.fg's four methods, s0, and
    transition_marks(), so it works for any FGBase subclass.
    """
    import matplotlib.pyplot as plt

    grid = build_example_grid()
    fg = grid.fg
    r2d = np.rad2deg

    v = np.linspace(-np.pi/2, np.pi/2, 1201)
    dfdv = np.array([fg.dfdv(vv) for vv in v])
    dgdv = np.array([fg.dgdv(vv) for vv in v])
    f_v  = np.array([fg.f(vv) for vv in v])
    g_v  = np.array([fg.g(vv) for vv in v])

    # ordinate rate -> meridional spacing in degrees per row
    to_deg_per_row = lambda d, w: r2d(np.abs(d) * 2.0/(1.0 + w**2) * grid.dvdj)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax = axes[0]
    ax.plot(r2d(v), dfdv, label="df/dv", lw=1.6)
    ax.plot(r2d(v), dgdv, label="dg/dv", lw=1.6)
    ax.axhline(0.0, color="0.7", lw=0.8)
    s0 = fg.s0
    ax.plot([0, 0], [s0, -s0], "k.", ms=8, zorder=5)
    ax.annotate(f"  f'(0) = +s0 = {s0:.4f}", (0, s0), fontsize=8, va="bottom")
    ax.annotate(f"  g'(0) = -s0", (0, -s0), fontsize=8, va="top")
    ax.set_ylabel("d/dv  [stereographic radius per rad of v]")
    ax.set_title("Shape derivatives (what the boundary conditions pin)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.plot(r2d(v), to_deg_per_row(dfdv, f_v), label="f branch (lon 270 / 90)", lw=1.6)
    ax.plot(r2d(v), to_deg_per_row(dgdv, g_v), label="g branch (lon 90)", lw=1.6)
    ax.axhline(r2d(fg.s0 * grid.dvdj), color="0.5", ls=":", lw=1.0,
               label="equatorial resolution")
    ax.set_ylabel("meridional spacing  [deg / row]")
    ax.set_xlabel("v  [deg]")
    ax.set_title("Same thing as grid resolution (paper Fig. 2)")
    ax.legend(loc="upper left", fontsize=9)

    for ax in axes:
        ax.axvspan(r2d(v[0]), 0.0, color="0.9", zorder=0)
        ax.annotate("not used by the mesh\n(lat-lon patch)", (r2d(v[0]) * 0.95, ax.get_ylim()[1]),
                    fontsize=7, color="0.4", va="top")
        for vt, name in fg.transition_marks():
            ax.axvline(r2d(vt), color="0.3", ls="--", lw=0.7, alpha=0.5)
        ax.axvline(90.0, color="k", lw=0.8, alpha=0.6)
        ax.grid(alpha=0.25)

    for vt, name in fg.transition_marks():
        axes[0].annotate(name, (r2d(vt), axes[0].get_ylim()[0]), fontsize=6,
                         rotation=90, va="bottom", ha="right", color="0.3")
    axes[0].annotate("mesh pole", (90.0, axes[0].get_ylim()[0]), fontsize=7,
                     rotation=90, va="bottom", ha="right")

    fig.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=110, bbox_inches="tight")
        print("Wrote plot: ", output_file)


if __name__ == "__main__":

    print("Plotting grid...")
    test_plot_fg_derivatives(output_file="figure_fg_derivative.png")
    test_plot_grid_naive()
    test_plot_grid()
    test_output_SCRIP_file()

