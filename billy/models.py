import numpy as np
import exoplanet as xo

def sinusoid_model(params, t):
    A = params[0]
    ω = params[1]
    φ = params[2]
    return A * np.sin(ω*t + φ)

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
