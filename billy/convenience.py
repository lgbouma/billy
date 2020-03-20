import numpy as np, pandas as pd
import collections
from collections import OrderedDict
from astropy.io import fits

def chisq(y_mod, y_obs, y_err):
    return np.sum( (y_mod - y_obs )**2 / y_err**2 )


def bic(chisq, k, n):
    """
    BIC = χ^2 + k log n, for k the number of free parameters, and n the
    number of data points.
    """
    return chisq + k*np.log(n)


def get_bic(m, ydict):

    y_obs = ydict['y_obs']
    y_err = m.y_err
    y_mod = ydict['y_mod_tra'] + ydict['y_mod_orb'] + ydict['y_mod_rot']

    χ2 = chisq(y_mod, y_obs, y_err)

    k = None
    if m.modelid == 'transit_1sincosPorb_1sincosProt':
        k = 13
    if m.modelid == 'transit_2sincosPorb_1sincosProt':
        k = 15
    if m.modelid == 'transit_1sincosPorb_2sincosProt':
        k = 15
    if m.modelid == 'transit_2sincosPorb_2sincosProt':
        k = 17
    if k is None:
        raise NotImplementedError

    n = len(y_obs)
    BIC = bic( χ2, k, n )

    dof = n-k

    msg = (
        '{}: χ2 = {:.1f}, redχ2 = {:.2f}, BIC = {:.1f}'.
        format(m.modelid, χ2, χ2/dof, BIC)
    )
    print(42*'=')
    print(msg)
    print(42*'=')


def flatten(l):
    for el in l:
        if (
            isinstance(el, collections.Iterable) and
            not isinstance(el, (str, bytes))
        ):
            yield from flatten(el)
        else:
            yield el


def get_ptfo_data(cdips=1, spoc=0):

    datadir = '/Users/luke/Dropbox/proj/billy/data/PTFO_8-8695'
    if cdips:
        # data: list of fits records
        lcfiles = [
            '/Users/luke/Dropbox/proj/billy/data/PTFO_8-8695/'
            'hlsp_cdips_tess_ffi_gaiatwo0003222255959210123904-0006_tess_v01_llc.fits'
        ]
    if spoc:
        lcfiles = [
            '/Users/luke/Dropbox/proj/billy/data/PTFO_8-8695/'
            'tess2018349182459-s0006-0000000264461976-0126-s/'
            'tess2018349182459-s0006-0000000264461976-0126-s_lc.fits'
        ]

    data = []
    for f in lcfiles:
        hdul = fits.open(f)
        data.append(hdul[1].data)

    return data


def initialize_ptfo_prior_d(x_obs, modelcomponents):

    P_orb = 0.4485 # +/- 1e-3
    t0_orb = 2458468.63809577 - 2457000 - 1468.2  # non-zero means phase offset.

    P_rot = 0.49845 # +/- 1e-3
    t0_rot = 0.14 # np.nanmin(x_obs) + 0.1 # initial guess

    prior_d = OrderedDict()

    for modelcomponent in modelcomponents:

        if 'transit' in modelcomponent:
            prior_d['period'] = P_orb
            prior_d['t0'] = t0_orb
            prior_d['r'] = 0.11 # -> 1.3% dip
            prior_d['b'] = 0.5  # initialize for broad prior
            prior_d['u'] = [0.30,0.35] # 0.39Msun, 1.39Rsun -> logg=3.74. Teff 3500K -> Claret18
            prior_d['mean'] = 0

        if 'sincos' in modelcomponent:

            N_harmonics = int(modelcomponent[0])
            for ix in range(N_harmonics):

                if 'Porb' in modelcomponent:
                    k = 'orb'
                elif 'Prot' in modelcomponent:
                    k = 'rot'
                else:
                    msg = 'expected Porb or Prot for freq specification'
                    raise NotImplementedError(msg)

                if k == 'rot':
                    if ix == 0:
                        prior_d['A{}{}'.format(k,ix)] = 3e-2
                        prior_d['B{}{}'.format(k,ix)] = 3e-2
                    elif ix == 1:
                        prior_d['A{}{}'.format(k,ix)] = 1e-2
                        prior_d['B{}{}'.format(k,ix)] = 1e-2
                    else:
                        raise NotImplementedError

                if k == 'orb':
                    if ix == 0:
                        prior_d['A{}{}'.format(k,ix)] = 1e-2
                        prior_d['B{}{}'.format(k,ix)] = 1e-2
                    elif ix == 1:
                        prior_d['A{}{}'.format(k,ix)] = 1e-2
                        prior_d['B{}{}'.format(k,ix)] = -1e-2
                    else:
                        raise NotImplementedError

                if 'Porb' in modelcomponent:
                    phi = 2*np.pi * t0_orb / P_orb
                elif 'Prot' in modelcomponent:
                    phi = 2*np.pi * t0_rot / P_rot
                prior_d['phi{}'.format(k)] = phi

                if 'Porb' in modelcomponent:
                    omega = 2*np.pi / P_orb
                elif 'Prot' in modelcomponent:
                    omega = 2*np.pi / P_rot
                prior_d['omega{}'.format(k)] = omega

    return prior_d
