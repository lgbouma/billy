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
                           '20200428_v0')

    assert modelid in [
        'transit_2sincosPorb_2sincosProt',
        'transit_2sincosPorb_3sincosProt'
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

        df = pm.summary(
            m.trace, varnames=list(prior_d.keys()),
            round_to=10, kind='stats'
        )

        df.to_csv(summarypath, index=True)

    else:
        df = pd.read_csv(summarypath, index_col=0)

    srows = ['period', 't0', 'r', 'b', 'u[0]', 'u[1]', 'mean',
             'omegaorb', 'Aorb0', 'Borb0', 'Aorb1', 'Borb1',
             'phirot', 'omegarot', 'Arot0', 'Brot0', 'Arot1', 'Brot1']
    if modelid == 'transit_2sincosPorb_3sincosProt':
        srows = ['period', 't0', 'r', 'b', 'u[0]', 'u[1]', 'mean',
                 'omegaorb', 'Aorb0', 'Borb0', 'Aorb1', 'Borb1',
                 'phirot', 'omegarot', 'Arot0', 'Brot0', 'Arot1', 'Brot1',
                 'Arot2', 'Brot2']

    df = df.loc[srows]

    pr = {
        'period': normal_str(mu=prior_d['period'], sd=1e-3,
                             fmtstr='({:.4f}; {:.4f})'),
        't0': normal_str(mu=prior_d['t0'], sd=0.002, fmtstr='({:.6f}; {:.4f})'),
        'r': normal_str(mu=prior_d['r'], sd=0.10*prior_d['r'], fmtstr='({:.4f}; {:.4f})'),
        'b': r'$\mathcal{U}(0; 1+R_{\mathrm{p}}/R_\star)$',
        'u[0]': '(1)',
        'u[1]': '(1)',
        'mean': uniform_str(lower=prior_d['mean']-1e-2,
                            upper=prior_d['mean']+1e-2),
        'omegaorb': '$2\pi/P_{\mathrm{s}}$',
        'Aorb0': uniform_str(lower=-2*np.abs(prior_d['Aorb0']),
                             upper=2*np.abs(prior_d['Aorb0'])),
        'Borb0': uniform_str(lower=-2*np.abs(prior_d['Borb0']),
                             upper=2*np.abs(prior_d['Borb0'])),
        'Aorb1': uniform_str(lower=-2*np.abs(prior_d['Aorb1']),
                             upper=2*np.abs(prior_d['Aorb1'])),
        'Borb1': uniform_str(lower=-2*np.abs(prior_d['Borb1']),
                             upper=2*np.abs(prior_d['Borb1'])),
        'phirot': uniform_str(lower=prior_d['phirot']-np.pi/8,
                              upper=prior_d['phirot']+np.pi/8, fmtstr='({:.4f}; {:.4f})'),
        'omegarot': normal_str(mu=prior_d['omegarot'],
                               sd=0.01*prior_d['omegarot'], fmtstr='({:.4f}; {:.4f})'),
        'Arot0': uniform_str(lower=-2*np.abs(prior_d['Arot0']),
                             upper=2*np.abs(prior_d['Arot0'])),
        'Brot0': uniform_str(lower=-2*np.abs(prior_d['Brot0']),
                             upper=2*np.abs(prior_d['Brot0'])),
        'Arot1': uniform_str(lower=-2*np.abs(prior_d['Arot1']),
                             upper=2*np.abs(prior_d['Arot1'])),
        'Brot1': uniform_str(lower=-2*np.abs(prior_d['Brot1']),
                             upper=2*np.abs(prior_d['Brot1']))
    }
    if modelid == 'transit_2sincosPorb_3sincosProt':
        pr['Arot2'] = uniform_str(lower=-2*np.abs(prior_d['Arot2']),
                                  upper=2*np.abs(prior_d['Arot2']))
        pr['Brot2'] = uniform_str(lower=-2*np.abs(prior_d['Brot2']),
                                  upper=2*np.abs(prior_d['Brot2']))

    # round everything. requires a double transpose because df.round
    # operates column-wise
    if modelid == 'transit_2sincosPorb_2sincosProt':
        round_precision = [7, 7, 5, 4, 3, 3, 6,
                           5, 6, 6, 6, 6,
                           5, 6, 6, 6, 6, 6]
    elif modelid == 'transit_2sincosPorb_3sincosProt':
        round_precision = [7, 7, 5, 4, 3, 3, 6,
                           5, 6, 6, 6, 6,
                           5, 6, 6, 6, 6, 6, 6, 6]
    else:
        raise NotImplementedError

    df = df.T.round(
        decimals=dict(
            zip(df.index, round_precision)
        )
    ).T

    df['priors'] = list(pr.values())

    df = df[
        ['priors', 'mean', 'sd', 'hpd_3%', 'hpd_97%']
    ]


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


def uniform_str(lower, upper, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{U}'+'({}; {})$'.format(lower, upper)
    else:
        return '$\\mathcal{U}'+'{}$'.format(fmtstr).format(lower, upper)


if __name__ == "__main__":

    main('transit_2sincosPorb_2sincosProt')
