import numpy as np, matplotlib.pyplot as plt, pandas as pd
import pickle, os, corner
import exoplanet as xo

def main():
    test_exoplanet_transit()

def test_exoplanet_transit():

    # The light curve calculation requires an orbit
    orbit = xo.orbits.KeplerianOrbit(period=3.456)
    texp = 30/(60*24)

    # Compute a limb-darkened light curve using starry
    t = np.linspace(-0.1, 0.1, 1000)
    u = [0.3, 0.2]
    light_curve = (
        xo.LimbDarkLightCurve(u)
        .get_light_curve(orbit=orbit, r=0.1, t=t, texp=texp)
        .eval()
    )
    # Note: the `eval` is needed because this is using Theano in
    # the background

    plt.plot(t, light_curve, color="C0", lw=2)
    plt.ylabel("relative flux")
    plt.xlabel("time [days]")
    _ = plt.xlim(t.min(), t.max())
    plt.savefig('../results/test_results/test_exoplanet_transit.png',
                bbox_inches='tight')
    plt.close('all')

if __name__ == "__main__":
    main()
