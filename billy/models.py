import numpy as np
import exoplanet as xo

def sin_model(params, t):
    A = params[0]
    ω = params[1]
    φ = params[2]
    return A * np.sin(ω*t + φ)


def cos_model(params, t):
    B = params[0]
    ω = params[1]
    φ = params[2]
    return B * np.cos(ω*t + φ)


def transit_model(params, t, texp=30/(60*24), mstar=1, rstar=1):
    period = params[0]
    t0 = params[1]
    r = params[2]
    b = params[3]
    u = params[4]
    mean = params[5]

    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, mstar=mstar,
                                     rstar=rstar)

    return (
        mean +
        xo.LimbDarkLightCurve(u)
        .get_light_curve(orbit=orbit, r=r, t=t, texp=texp)
        .eval().flatten()
    )


def linear_model(params, x, x_occ=None):
    """
    Linear model. Parameters (t0, P).
    Must pass transit times.

    If x_occ is none, returns model t_tra array.
    If x_occ is a numpy array, returns tuple of model t_tra and t_occ arrays.
    """
    t0, period = params
    if not isinstance(x_occ,np.ndarray):
        return t0 + period*x
    else:
        return t0 + period*x, t0 + period/2 + period*x_occ


def quadratic_model(params, x, x_occ=None):
    """
    Quadratic model. Parameters (t0, P, 0.5dP/dE).
    Must pass transit times.

    If x_occ is none, returns model t_tra array.
    If x_occ is a numpy array, returns tuple of model t_tra and t_occ arrays.
    """
    t0, period, half_dP_dE = params
    if not isinstance(x_occ,np.ndarray):
        return t0 + period*x + half_dP_dE*x**2
    else:
        return (t0 + period*x + half_dP_dE*x**2,
                t0 + period/2 + period*x_occ + half_dP_dE*x_occ**2
               )
