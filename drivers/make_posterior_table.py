import os, re
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join
from itertools import product

from billy.modelfitter import ModelFitter, ModelParser
import billy.plotting as bp
from billy.convenience import (
    get_clean_ptfo_data, get_ptfo_data, initialize_ptfo_prior_d, get_bic
)
from billy import __path__

def main(modelid):

    OVERWRITE = 0
    REALID = 'PTFO_8-8695'
    RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
    PLOTDIR = os.path.join(RESULTSDIR, '{}_results'.format(REALID),
                           '20200513_v0')

    assert modelid in [
        'transit_2sincosPorb_2sincosProt',
        'transit_2sincosPorb_3sincosProt',
        'transit_3sincosPorb_2sincosProt'
    ]

    summarypath = os.path.join(
        PLOTDIR, 'posterior_table_raw_{}.csv'.format(modelid)
    )

    pklpath = os.path.join(
        os.path.expanduser('~'), 'local', 'billy',
        '{}_model_{}.pkl'.format(REALID, modelid)
    )
    np.random.seed(42)

    x_obs, y_obs, y_err = get_clean_ptfo_data()

    mp = ModelParser(modelid)
    prior_d = initialize_ptfo_prior_d(x_obs, mp.modelcomponents)

    if not os.path.exists(summarypath):

        m = ModelFitter(modelid, x_obs, y_obs, y_err, prior_d, plotdir=PLOTDIR,
                        pklpath=pklpath, overwrite=OVERWRITE)

        # NOTE: you could pass "varnames = varnames", but don't to enable
        # derived parameter collection
        # varnames = list(prior_d.keys())

        df = pm.summary(
            m.trace,
            round_to=10, kind='stats'
        )

        df.to_csv(summarypath, index=True)

    else:
        df = pd.read_csv(summarypath, index_col=0)

    srows = [
        # fitted params
        'period', 't0', 'r', 'b', 'u[0]', 'u[1]', 'mean', 'r_star', 'm_star',
        'Aorb0', 'Borb0', 'Aorb1', 'Borb1', 'phirot', 'omegarot', 'Arot0',
        'Brot0', 'Arot1', 'Brot1',
        # derived params
        'omegaorb', 'rhostar', 'r_planet', 'a_Rs'
    ]
    if modelid == 'transit_2sincosPorb_3sincosProt':
        srows = [
            'period', 't0', 'r', 'b', 'u[0]', 'u[1]', 'mean', 'r_star',
            'm_star', 'Aorb0', 'Borb0', 'Aorb1', 'Borb1', 'phirot', 'omegarot',
            'Arot0', 'Brot0', 'Arot1', 'Brot1', 'Arot2', 'Brot2',
            'omegaorb', 'rhostar', 'r_planet', 'a_Rs'
        ]
    if modelid == 'transit_3sincosPorb_2sincosProt':
        srows = [
            'period', 't0', 'r', 'b', 'u[0]', 'u[1]', 'mean', 'r_star',
            'm_star', 'Aorb0', 'Borb0', 'Aorb1', 'Borb1', 'Aorb2', 'Borb2', 'phirot', 'omegarot',
            'Arot0', 'Brot0', 'Arot1', 'Brot1',
            'omegaorb', 'rhostar', 'r_planet', 'a_Rs'
        ]

    df = df.loc[srows]

    from billy.convenience import (
        MSTAR_VANEYKEN, MSTAR_STDEV, RSTAR_VANEYKEN, RSTAR_STDEV
    )

    pr = {
        'period': normal_str(mu=prior_d['period'], sd=1e-3,
                             fmtstr='({:.4f}; {:.4f})'),
        't0': normal_str(mu=prior_d['t0'], sd=0.002, fmtstr='({:.6f}; {:.4f})'),
        'r': normal_str(mu=prior_d['r'], sd=0.10*prior_d['r'], fmtstr='({:.4f}; {:.4f})'),
        'b': r'$\mathcal{U}(0; 1+R_{\mathrm{p}}/R_\star)$',
        'u[0]': '(2)',
        'u[1]': '(2)',
        'mean': uniform_str(lower=prior_d['mean']-1e-2,
                            upper=prior_d['mean']+1e-2),
        'r_star': truncnormal_str(mu=RSTAR_VANEYKEN, sd=RSTAR_STDEV,
                                  fmtstr='({:.2f}; {:.2f})'),
        'm_star': truncnormal_str(mu=MSTAR_VANEYKEN, sd=MSTAR_STDEV,
                                  fmtstr='({:.2f}; {:.2f})'),
        'Aorb0': uniform_str(lower=-2*np.abs(prior_d['Aorb0']),
                             upper=2*np.abs(prior_d['Aorb0'])),
        'Borb0': uniform_str(lower=-2*np.abs(prior_d['Borb0']),
                             upper=2*np.abs(prior_d['Borb0'])),
        'Aorb1': uniform_str(lower=-2*np.abs(prior_d['Aorb1']),
                             upper=2*np.abs(prior_d['Aorb1'])),
        'Borb1': uniform_str(lower=-2*np.abs(prior_d['Borb1']),
                             upper=2*np.abs(prior_d['Borb1'])),
    }
    if modelid == 'transit_3sincosPorb_2sincosProt':
        pr['Aorb2'] = uniform_str(lower=-2*np.abs(prior_d['Aorb2']),
                                  upper=2*np.abs(prior_d['Aorb2']))
        pr['Borb2'] = uniform_str(lower=-2*np.abs(prior_d['Borb2']),
                                  upper=2*np.abs(prior_d['Borb2']))

    pr['phirot'] = uniform_str(lower=prior_d['phirot']-np.pi/8,
                               upper=prior_d['phirot']+np.pi/8,
                               fmtstr='({:.4f}; {:.4f})')
    pr['omegarot'] = normal_str(mu=prior_d['omegarot'],
                                sd=0.01*prior_d['omegarot'],
                                fmtstr='({:.4f}; {:.4f})')
    pr['Arot0'] = uniform_str(lower=-2*np.abs(prior_d['Arot0']),
                              upper=2*np.abs(prior_d['Arot0']))
    pr['Brot0'] = uniform_str(lower=-2*np.abs(prior_d['Brot0']),
                              upper=2*np.abs(prior_d['Brot0']))
    pr['Arot1'] = uniform_str(lower=-2*np.abs(prior_d['Arot1']),
                              upper=2*np.abs(prior_d['Arot1']))
    pr['Brot1'] = uniform_str(lower=-2*np.abs(prior_d['Brot1']),
                              upper=2*np.abs(prior_d['Brot1']))

    if modelid == 'transit_2sincosPorb_3sincosProt':
        pr['Arot2'] = uniform_str(lower=-2*np.abs(prior_d['Arot2']),
                                  upper=2*np.abs(prior_d['Arot2']))
        pr['Brot2'] = uniform_str(lower=-2*np.abs(prior_d['Brot2']),
                                  upper=2*np.abs(prior_d['Brot2']))
    pr['omegaorb'] = '--'
    pr['rhostar'] = '--'
    pr['r_planet'] = '--'
    pr['a_Rs'] = '--'

    # round everything. requires a double transpose because df.round
    # operates column-wise
    if modelid == 'transit_2sincosPorb_2sincosProt':
        round_precision = [7, 7, 5, 4, 3, 3, 6,
                           2, 2,
                           6, 6, 6, 6,
                           5, 6, 6, 6, 6, 6,
                           5, 2, 2, 2
                           ]
    elif modelid == 'transit_3sincosPorb_2sincosProt':
        round_precision = [7, 7, 5, 4, 3, 3, 6,
                           2, 2,
                           6, 6, 6, 6, 6, 6,
                           5, 6, 6, 6, 6, 6,
                           5, 2, 2, 2]
    elif modelid == 'transit_2sincosPorb_3sincosProt':
        round_precision = [7, 7, 5, 4, 3, 3, 6,
                           2, 2,
                           6, 6, 6, 6,
                           5, 6, 6, 6, 6, 6, 6, 6,
                           5, 2, 2, 2]
    else:
        raise NotImplementedError

    df = df.T.round(
        decimals=dict(
            zip(df.index, round_precision)
        )
    ).T

    df['priors'] = list(pr.values())

    # units
    ud = {
        'period': 'd',
        't0': 'd',
        'r': '--',
        'b': '--',
        'u[0]': '--',
        'u[1]': '--',
        'mean': '--',
        'r_star': r'$R_\odot$',
        'm_star': r'$M_\odot$',
        'Aorb0': '--',
        'Borb0': '--',
        'Aorb1': '--',
        'Borb1': '--'
    }
    if modelid == 'transit_3sincosPorb_2sincosProt':
        ud['Aorb2'] = '--'
        ud['Borb2'] = '--'

    ud['phirot'] = 'rad'
    ud['omegarot'] = 'rad$\ $d$^{-1}$'
    ud['Arot0'] = '--'
    ud['Brot0'] = '--'
    ud['Arot1'] = '--'
    ud['Brot1'] = '--'

    if modelid == 'transit_2sincosPorb_3sincosProt':
        ud['Arot2'] = '--'
        ud['Brot2'] = '--'

    ud['omegaorb'] = 'rad$\ $d$^{-1}$'
    ud['rhostar'] = 'g$\ $cm$^{-3}$'
    ud['r_planet'] = '$R_{\mathrm{Jup}}$'
    ud['a_Rs'] = '--'

    df['units'] = list(ud.values())

    df = df[
        ['units', 'priors', 'mean', 'sd', 'hpd_3%', 'hpd_97%']
    ]

    latexparams = [
        r"$P_{\rm s}$",
        r"$t_{\rm s}^{(1)}$",
        r"$R_{\rm p}/R_\star$",
        "$b$",
        "$u_1$",
        "$u_2$",
        "Mean",
        "$R_\star$",
        "$M_\star$",
        "$A_{\mathrm{s}0}$",
        "$B_{\mathrm{s}0}$",
        "$A_{\mathrm{s}1}$",
        "$B_{\mathrm{s}1}$",
        "$A_{\mathrm{s}2}$",
        "$B_{\mathrm{s}2}$",
        r"$\phi_{\rm \ell}$",
        r"$\omega_{\rm \ell}$",
        "$A_{\mathrm{\ell}0}$",
        "$B_{\mathrm{\ell}0}$",
        "$A_{\mathrm{\ell}1}$",
        "$B_{\mathrm{\ell}1}$",
        # derived
        r"$\omega_{\rm s}$",
        r"$\rho_\star$",
        r"$R_{\rm p}$",
        "$a/R_\star$"
    ]
    df.index = latexparams

    outpath = os.path.join(PLOTDIR,
                           'posterior_table_clean_{}.csv'.format(modelid))
    df.to_csv(outpath, float_format='%.12f')
    print('made {}'.format(outpath))

    # df.to_latex is dumb with float formatting.
    outpath = os.path.join(PLOTDIR,
                           'posterior_table_clean_{}.tex'.format(modelid))
    df.to_csv(outpath, sep=',', line_terminator=' \\\\\n',
              float_format='%.12f')

    with open(outpath, 'r') as f:
        lines = f.readlines()

    for ix, l in enumerate(lines):

        # replace commas with latex ampersands
        thisline = deepcopy(l.replace(',', ' & '))

        # replace quotes with nada
        thisline = thisline.replace('"', '')

        # replace }0 with },0
        thisline = thisline.replace('}0', '},0')
        thisline = thisline.replace('}1', '},1')
        thisline = thisline.replace('}2', '},2')

        if ix == 0:
            lines[ix] = thisline
            continue

        # iteratively replace stupid trailing zeros with whitespace
        while re.search("0{2,10}\ ", thisline) is not None:
            r = re.search("0{2,10}\ ", thisline)
            thisline = thisline.replace(
                thisline[r.start():r.end()],
                ' '
            )

        lines[ix] = thisline

    with open(outpath, 'w') as f:
        f.writelines(lines)

    print('made {}'.format(outpath))


def normal_str(mu, sd, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{N}'+'({}; {})$'.format(mu, sd)
    else:
        return '$\\mathcal{N}'+'{}$'.format(fmtstr).format(mu, sd)


def truncnormal_str(mu, sd, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{T}'+'({}; {})$'.format(mu, sd)
    else:
        return '$\\mathcal{T}'+'{}$'.format(fmtstr).format(mu, sd)


def uniform_str(lower, upper, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{U}'+'({}; {})$'.format(lower, upper)
    else:
        return '$\\mathcal{U}'+'{}$'.format(fmtstr).format(lower, upper)


if __name__ == "__main__":

    main('transit_3sincosPorb_2sincosProt')
