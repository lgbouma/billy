"""
Fit data for "transit_NsincosPorb_NsincosProt" model.
"""
import os
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

    make_threadsafe = 0

    traceplot = 0
    sampleplot = 0
    cornerplot = 0
    splitsignalplot = 1 if 'Porb' in modelid and 'Prot' in modelid else 0

    OVERWRITE = 1
    REALID = 'PTFO_8-8695'
    RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
    # NOTE: 20200513 used in paper; 20201016 for generating cute money plot
    #PLOTDIR = os.path.join(RESULTSDIR, '{}_results'.format(REALID),
    #                       '20200513_v0')
    PLOTDIR = os.path.join(RESULTSDIR, '{}_results'.format(REALID),
                           '20201016_v0')

    ##########

    print(42*'#')
    print(modelid)
    print(42*'#')

    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)
    pklpath = os.path.join(
        os.path.expanduser('~'), 'local', 'billy',
        '{}_model_{}.pkl'.format(REALID, modelid)
    )
    np.random.seed(42)

    x_obs, y_obs, y_err = get_clean_ptfo_data()

    mp = ModelParser(modelid)
    prior_d = initialize_ptfo_prior_d(x_obs, mp.modelcomponents)
    m = ModelFitter(modelid, x_obs, y_obs, y_err, prior_d, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=OVERWRITE)

    print(pm.summary(m.trace, varnames=list(prior_d.keys())))

    if make_threadsafe:
        pass

    else:
        if traceplot:
            outpath = join(PLOTDIR, '{}_{}_traceplot.png'.format(REALID, modelid))
            bp.plot_traceplot(m, outpath)

        if sampleplot:
            outpath = join(PLOTDIR, '{}_{}_sampleplot.png'.format(REALID, modelid))
            bp.plot_sampleplot(m, outpath, N_samples=10)

        if splitsignalplot:
            do_post = 0
            do_map = 1
            if do_post:
                outpath = join(PLOTDIR, '{}_{}_splitsignalpost.png'.format(REALID, modelid))
                ydict = bp.plot_splitsignal_post(m, outpath)
                outpath = join(PLOTDIR, '{}_{}_phasefoldpost.png'.format(REALID, modelid))
                bp.plot_phasefold_post(m, ydict, outpath)
            if do_map:
                outpath = join(PLOTDIR, '{}_{}_splitsignalmap_i.png'.format(REALID, modelid))
                ydict = bp.plot_splitsignal_map(m, outpath, part='i')
                outpath = join(PLOTDIR, '{}_{}_splitsignalmap_ii.png'.format(REALID, modelid))
                ydict = bp.plot_splitsignal_map(m, outpath, part='ii')
                outpath = join(PLOTDIR, '{}_{}_splitsignalmap_periodogram.png'.format(REALID, modelid))
                if not os.path.exists(outpath) or m.OVERWRITE:
                    bp.plot_splitsignal_map_periodogram(ydict, outpath)
                outpath = join(PLOTDIR, '{}_{}_phasefoldmap.png'.format(REALID, modelid))
                bp.plot_phasefold_map(m, ydict, outpath)
                outpath = join(PLOTDIR, '{}_{}_phasefoldmap_shrinkratio.png'.format(REALID, modelid))
                bp.plot_phasefold_map(m, ydict, outpath, shrinkratio=1)
                outpath = join(PLOTDIR, f'{REALID}_{modelid}_phasefoldmap_shrinkratio_singlepanel.png')
                bp.plot_phasefold_map_singlepanel(m, ydict, outpath, shrinkratio=1)
                outpath = join(PLOTDIR, f'{REALID}_{modelid}_phasefoldmap_shrinkratio_singlepanel_gridoff.png')
                bp.plot_phasefold_map_singlepanel(m, ydict, outpath, shrinkratio=1, gridoff=1)
                get_bic(m, ydict, PLOTDIR)

        if cornerplot:
            prior_d.pop('omegaorb', None) # not sampled; only used in data generation
            prior_d.pop('phiorb', None) # not sampled; only used in data generation
            outpath = join(PLOTDIR, '{}_{}_cornerplot.png'.format(REALID, modelid))
            bp.plot_cornerplot(prior_d, m, outpath)


if __name__ == "__main__":

    DEBUG = 1

    if DEBUG:
        main('transit_2sincosPorb_3sincosProt')

    else:
        for N, M in product(range(1,4), range(1,4)):
            main('transit_{}sincosPorb_{}sincosProt'.format(N,M))
        # DEPRECATED
        # main('transit_2sincosPorb_2sincosProt')
        # main('transit_1sincosPorb_2sincosProt')
        # main('transit_2sincosPorb_1sincosProt')
