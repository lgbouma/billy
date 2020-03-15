"""
Given a modelid specification string, e.g.,
"transit_NsincosPorb_NsincosProt", construct appropriate synthetic data, and
then fit it.
"""

import numpy as np, matplotlib.pyplot as plt, pymc3 as pm
import os, corner
from pymc3.backends.tracetab import trace_to_dataframe

from billy.fakedata import FakeDataGenerator
from billy.modelfitter import ModelFitter
from billy.convenience import flatten as bflatten

modelid = 'transit'
#modelid = 'transit_2sincosPorb_2sincosProt'
traceplot = 0
cornerplot = 1

np.random.seed(42)

f = FakeDataGenerator(modelid)
m = ModelFitter(modelid, f.x_obs, f.y_obs, f.y_err, f.true_d)

if traceplot:
    # trace plot from PyMC3
    outpath = '../results/driver_results/test_{}_traceplot.png'.format(modelid)
    if not os.path.exists(outpath):
        plt.figure(figsize=(7, 7))
        pm.traceplot(m.trace[100:])
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close('all')

if cornerplot:
    # corner
    print(pm.summary(m.trace, varnames=list(f.true_d.keys())))

    trace_df = trace_to_dataframe(m.trace, varnames=list(f.true_d.keys()))

    truths = [f.true_d[k] for k in f.true_d.keys()]
    truths = list(bflatten(truths))
    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        truths=truths)
    fig.savefig('../results/driver_results/test_{}_corner.png'.format(modelid))
