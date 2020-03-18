"""
Fit data for "transit_NsincosPorb_NsincosProt" model.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join

from billy.modelfitter import ModelFitter, ModelParser
import billy.plotting as bp
from billy.convenience import get_ptfo_data, initialize_ptfo_prior_d
from billy import __path__

from astrobase.lcmath import time_bin_magseries_with_errs

def main(modelid):

    REALID = 'PTFO_8-8695'

    traceplot = 0
    sampleplot = 1
    cornerplot = 1
    splitsignalplot = 1 if 'Porb' in modelid and 'Prot' in modelid else 0

    RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
    PLOTDIR = os.path.join(RESULTSDIR, '{}_results'.format(REALID))
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)
    pklpath = os.path.join(
        os.path.expanduser('~'), 'local', 'billy',
        '{}_model_{}.pkl'.format(REALID, modelid)
    )

    np.random.seed(42)

    # get data. quality cut and remove weird end points. bin to 10 minutes, to
    # speed fitting (which is linear in time).
    d = get_ptfo_data(cdips=0, spoc=1)[0]
    quality = d.QUALITY
    x_obs = d.TIME - 1468.2
    sel = (quality == 0) & (x_obs < 20.1)
    x_obs = d.TIME[sel] - 1468.2
    y_obs = (d.PDCSAP_FLUX[sel] / np.nanmedian(d.PDCSAP_FLUX[sel])) - 1
    y_err = d.PDCSAP_FLUX_ERR[sel] / np.nanmedian(d.PDCSAP_FLUX[sel])
    binsize = 120*5 # 10 minutes
    bd = time_bin_magseries_with_errs(x_obs, y_obs, y_err, binsize=binsize,
                                      minbinelems=5)
    x_obs = bd['binnedtimes']
    y_obs = bd['binnedmags']
    y_err = bd['binnederrs']

    mp = ModelParser(modelid)
    prior_d = initialize_ptfo_prior_d(x_obs, mp.modelcomponents)
    m = ModelFitter(modelid, x_obs.astype(np.float64), y_obs.astype(np.float64),
                    y_err.astype(np.float64), prior_d, plotdir=PLOTDIR,
                    pklpath=pklpath)

    print(pm.summary(m.trace, varnames=list(prior_d.keys())))

    if traceplot:
        outpath = join(PLOTDIR, '{}_{}_traceplot.png'.format(REALID, modelid))
        bp.plot_traceplot(m, outpath)
    if sampleplot:
        outpath = join(PLOTDIR, '{}_{}_sampleplot.png'.format(REALID, modelid))
        bp.plot_sampleplot(m, outpath, N_samples=100)

    if splitsignalplot:
        do_post = 1
        do_map = 1
        if do_post:
            outpath = join(PLOTDIR, '{}_{}_splitsignalpost.png'.format(REALID, modelid))
            ydict = bp.plot_splitsignal_post(m, outpath)
            outpath = join(PLOTDIR, '{}_{}_phasefoldpost.png'.format(REALID, modelid))
            bp.plot_phasefold_post(m, ydict, outpath)
        if do_map:
            outpath = join(PLOTDIR, '{}_{}_splitsignalmap.png'.format(REALID, modelid))
            ydict = bp.plot_splitsignal_map(m, outpath)
            outpath = join(PLOTDIR, '{}_{}_phasefoldmap.png'.format(REALID, modelid))
            bp.plot_phasefold_map(m, ydict, outpath)
    if cornerplot:
        prior_d.pop('omegaorb', None) # not sampled; only used in data generation
        prior_d.pop('phiorb', None) # not sampled; only used in data generation
        outpath = join(PLOTDIR, '{}_{}_cornerplot.png'.format(REALID, modelid))
        bp.plot_cornerplot(prior_d, m, outpath)

if __name__ == "__main__":
    main('transit_2sincosPorb_1sincosProt')
    main('transit_2sincosPorb_2sincosProt')
