"""
Given a modelid specification string, e.g.,
"transit_NsincosPorb_NsincosProt", construct appropriate synthetic data, and
then fit it.
"""

import os
import numpy as np, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join

from billy.fakedata import FakeDataGenerator
from billy.modelfitter import ModelFitter
import billy.plotting as bp
from billy import __path__

RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
PLOTDIR = os.path.join(RESULTSDIR, 'synthetic_results')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

modelid = 'transit_1sincosPorb_1sincosProt'
traceplot = 0
sampleplot = 1
cornerplot = 1
splitsignalplot = 1 if 'Porb' in modelid and 'Prot' in modelid else 0

np.random.seed(42)

f = FakeDataGenerator(modelid, PLOTDIR)
m = ModelFitter(modelid, f.x_obs, f.y_obs, f.y_err, f.true_d, plotdir=PLOTDIR)

print(pm.summary(m.trace, varnames=list(f.true_d.keys())))

if traceplot:
    outpath = join(PLOTDIR, 'synthetic_{}_traceplot.png'.format(modelid))
    bp.plot_traceplot(m, outpath)
if splitsignalplot:
    outpath = join(PLOTDIR, 'synthetic_{}_splitsignal.png'.format(modelid))
    bp.plot_splitsignal(m, outpath)
if sampleplot:
    outpath = join(PLOTDIR, 'synthetic_{}_sampleplot.png'.format(modelid))
    bp.plot_sampleplot(m, outpath, N_samples=100)
if cornerplot:
    f.true_d.pop('omegaorb', None) # not sampled; only used in data generation
    f.true_d.pop('phiorb', None) # not sampled; only used in data generation
    outpath = join(PLOTDIR, 'synthetic_{}_cornerplot.png'.format(modelid))
    bp.plot_cornerplot(f, m, outpath)
