"""
Y ~ N( A0 sin(ω0t + φ0) + A1 sin(ω1t + φ1), σ^2).
Errors are treated as fixed and known observables.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner
from collections import OrderedDict
from pymc3.backends.tracetab import trace_to_dataframe
from billy.models import sinusoid_model
from billy.plotting import plot_test_data

#################
# generate data #
#################
modelid = 'multisinusoid'
size = 500
N_samples = 1000
true_sigma = 0.1
true_d = OrderedDict({'A_0': 1, 'omega_0': 2, 'phi_0': 0.3141,
                      'A_1': 0.2, 'omega_1': 1.8, 'phi_1': 0.5})

true_params_0 = [true_d[k] for k in ['A_0','omega_0','phi_0']]
true_params_1 = [true_d[k] for k in ['A_1','omega_1','phi_1']]
true_params = [true_d[k] for k in true_d.keys()]

# make fake data and add noise
np.random.seed(42)
x_obs = np.linspace(0, 10, size)
y_mod = (
    sinusoid_model(true_params_0, x_obs) +
    sinusoid_model(true_params_1, x_obs)
)
y_obs = y_mod + np.random.normal(scale=true_sigma, size=size)

plot_test_data(x_obs, y_obs, y_mod, modelid, outdir='../results/test_results/')

####################################
# fit a line and sample parameters #
####################################
pklpath = '../results/test_results/model_{}.pkl'.format(modelid)

if not os.path.exists(pklpath):
    with pm.Model() as model:
        sigma = true_sigma

        # Define priors
        A_0 = pm.Uniform('A_0', lower=0.5, upper=2)
        omega_0 = pm.Uniform('omega_0', lower=1.9, upper=2.1)
        phi_0 = pm.Uniform('phi_0', lower=0, upper=np.pi)

        A_1 = pm.Uniform('A_1', lower=0.1, upper=0.3)
        omega_1 = pm.Uniform('omega_1', lower=1.7, upper=1.9)
        phi_1 = pm.Uniform('phi_1', lower=0, upper=np.pi)

        # Define likelihood
        mu_model = (
            sinusoid_model([A_0, omega_0, phi_0], x_obs)
            +
            sinusoid_model([A_1, omega_1, phi_1], x_obs)
        )

        likelihood = pm.Normal('y', mu=mu_model,
                               sigma=sigma, observed=y_obs)

        # Inference!  draw posterior samples using NUTS sampling
        trace = pm.sample(N_samples, cores=16)

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
