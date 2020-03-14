"""
Y ~ N(A * np.sin(ω*t + φ_0), σ^2).
Errors are treated as an additive free parameter in the model.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner
from pymc3.backends.tracetab import trace_to_dataframe
from billy.models import sinusoid_model
from billy.plotting import plot_test_data

#################
# generate data #
#################
modelid = 'sinusoid_floaterr'
size = 200
true_d = {'A': 1, 'omega': 2, 'phi0': 0.3141, 'sigma': 0.2 }
true_params = [true_d['A'], true_d['omega'], true_d['phi0']]
true_sigma = true_d['sigma']

# make fake data and add noise
np.random.seed(42)
x_obs = np.linspace(0, 3, size)
y_mod = sinusoid_model(true_params, x_obs)
y_obs = y_mod + np.random.normal(scale=true_sigma, size=size)

plot_test_data(x_obs, y_obs, y_mod, modelid, outdir='../results/test_results/')

####################################
# fit a line and sample parameters #
####################################
pklpath = '../results/test_results/model_{}.pkl'.format(modelid)

if not os.path.exists(pklpath):
    with pm.Model() as model:
        # Define priors
        A = pm.Uniform('A', lower=0.1, upper=2)
        omega = pm.Normal('omega', mu=2, sigma=2)
        phi0 = pm.Uniform('phi0', lower=0, upper=np.pi)
        sigma = pm.HalfCauchy('sigma', beta=2, testval=1.)

        # Define likelihood
        likelihood = pm.Normal('y', mu=sinusoid_model([A, omega, phi0], x_obs),
                               sigma=sigma, observed=y_obs)

        # Inference!  draw 3000 posterior samples using NUTS sampling
        trace = pm.sample(3000, cores=16)

        map_estimate = pm.find_MAP(model=model)

    with open(pklpath, 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace,
                     'map_estimate': map_estimate}, buff)
else:
    d = pickle.load(open(pklpath, 'rb'))
    model, trace, map_estimate = d['model'], d['trace'], d['map_estimate']

##################
# analyze output #
##################
# trace plot from PyMC3
plt.figure(figsize=(7, 7))
pm.traceplot(trace[100:])
plt.tight_layout()
plt.savefig('../results/test_results/test_{}_traceplot.png'.format(modelid))
plt.close('all')

# corner
trace_df = trace_to_dataframe(trace)
truths = [true_d[k] for k in list(trace_df.columns)]
fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    title_kwargs={"fontsize": 12}, truths=truths)
fig.savefig('../results/test_results/test_{}_corner.png'.format(modelid))
