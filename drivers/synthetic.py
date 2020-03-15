"""
Given a modelid specification string, e.g.,
"transit_NsincosPorb_NsincosProt", construct appropriate synthetic data, and
then fit it.
"""

import numpy as np, matplotlib.pyplot as plt, pymc3 as pm

from billy.fakedata import FakeDataGenerator
from billy.modelfitter import ModelFitter
import billy.plotting as bp

modelid = 'transit_2sincosPorb_1sincosProt'
traceplot = 0
cornerplot = 1

np.random.seed(42)

f = FakeDataGenerator(modelid)
m = ModelFitter(modelid, f.x_obs, f.y_obs, f.y_err, f.true_d)

print(pm.summary(m.trace, varnames=list(f.true_d.keys())))

if traceplot:
    bp.plot_traceplot(m)
if cornerplot:
    bp.plot_cornerplot(f, m)
