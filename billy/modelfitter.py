import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os
from collections import OrderedDict

import exoplanet as xo

from billy import __path__
from billy.models import sin_model, cos_model, transit_model
from billy.plotting import plot_test_data, savefig
from billy.convenience import flatten as bflatten

RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')

class ModelParser:

    def __init__(self, modelid):
        self.initialize_model(self.modelid)

    def initialize_model(self, modelid):
        self.modelid = modelid
        self.modelcomponents = modelid.split('_')
        self.verify_modelcomponents()

    def verify_modelcomponents(self):

        validcomponents = ['transit']
        for i in range(5):
            validcomponents.append('{}sincosPorb'.format(i))
            validcomponents.append('{}sincosProt'.format(i))

        assert len(self.modelcomponents) >= 1

        for modelcomponent in self.modelcomponents:
            if modelcomponent not in validcomponents:
                errmsg = (
                    'Got modelcomponent {}. validcomponents include {}.'
                    .format(modelcomponent, validcomponents)
                )
                raise ValueError(errmsg)


class ModelFitter(ModelParser):
    """
    Given a modelid of the form "transit_NsincosPorb_NsincosProt", and observed
    x and y values (typically time and flux), construct and fit the model. In
    other words, run the inference.
    """

    def __init__(self, modelid, x_obs, y_obs, y_err, prior_d,
                 mstar=1, rstar=1, N_samples=1000, N_cores=16, N_chains=4):

        self.N_samples = N_samples
        self.N_cores = N_cores
        self.N_chains = N_chains

        self.x_obs = x_obs
        self.y_obs = y_obs
        self.y_err = y_err
        self.mstar = mstar
        self.rstar = rstar
        self.t_exp = np.nanmedian(x_obs)

        self.initialize_model(modelid)
        self.verify_inputdata()
        self.run_inference(prior_d)


    def verify_inputdata(self):
        assert len(self.x_obs) == len(self.y_obs)
        assert isinstance(self.x_obs, np.ndarray)
        assert isinstance(self.y_obs, np.ndarray)


    def run_inference(self, prior_d):

        pklpath = os.path.join(
            os.path.expanduser('~'), 'local', 'billy',
            'model_{}.pkl'.format(self.modelid)
        )

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1

        with pm.Model() as model:

            # Fixed data errors.
            sigma = self.y_err

            # Define priors and PyMC3 random variables to sample over.
            A_d, B_d, omega_d, phi_d = {}, {}, {}, {}
            for modelcomponent in self.modelcomponents:

                if 'transit' in modelcomponent:

                    mean = pm.Normal(
                        "mean", mu=0.0, sd=1.0, testval=prior_d['mean']
                    )

                    t0 = pm.Uniform(
                        "t0", lower=0, upper=prior_d['period'],
                        testval=prior_d['t0']
                    )

                    logP = pm.Normal(
                        "logP", mu=np.log(prior_d['period']), sd=0.1,
                        testval=np.log(prior_d['period'])
                    )
                    period = pm.Deterministic("period", pm.math.exp(logP))

                    u = xo.distributions.QuadLimbDark(
                        "u", testval=prior_d['u']
                    )

                    r = pm.Uniform(
                        "r", lower=prior_d['r']-1e-2, upper=prior_d['r']+1e-2,
                        testval=prior_d['r']

                    )

                    b = xo.distributions.ImpactParameter(
                        "b", ror=r, testval=prior_d['b']
                    )

                    orbit = xo.orbits.KeplerianOrbit(
                        period=period, t0=t0, b=b,
                        mstar=self.mstar, rstar=self.rstar
                    )
                    light_curve = (
                        mean +
                        xo.LimbDarkLightCurve(u).get_light_curve(
                            orbit=orbit, r=r, t=self.x_obs, texp=self.t_exp
                        )
                    )
                    pm.Deterministic("light_curve", light_curve)

                if 'sincos' in modelcomponent:
                    if 'Porb' in modelcomponent:
                        k = 'orb'
                    elif 'Prot' in modelcomponent:
                        k = 'rot'
                    else:
                        msg = 'expected Porb or Prot for freq specification'
                        raise NotImplementedError(msg)

                    omegakey = 'omega{}'.format(k)
                    omega_d[omegakey] = pm.Uniform(omegakey,
                                                   lower=prior_d[omegakey]-1e-2,
                                                   upper=prior_d[omegakey]+1e-2,
                                                   testval=prior_d[omegakey])

                    phikey = 'phi{}'.format(k)
                    phi_d[phikey] = pm.Uniform(phikey, lower=0, upper=np.pi,
                                               testval=prior_d[phikey])

                    N_harmonics = int(modelcomponent[0])
                    for ix in range(N_harmonics):

                        Akey = 'A{}{}'.format(k,ix)
                        Bkey = 'B{}{}'.format(k,ix)

                        A_d[Akey] = pm.Uniform(Akey, lower=prior_d[Akey]-1e-4,
                                                upper=prior_d[Akey]+1e-4,
                                                testval=prior_d[Akey])

                        B_d[Bkey] = pm.Uniform(Bkey, lower=prior_d[Bkey]-1e-4,
                                               upper=prior_d[Bkey]+1e-4,
                                               testval=prior_d[Bkey])

            harmonic_d = {**A_d, **B_d, **omega_d, **phi_d}

            # Build the likelihood

            if 'transit' not in self.modelcomponents:
                # NOTE: hacky implementation detail: I didn't now how else to
                # initialize an "empty" pymc3 random variable, so I assumed
                # here that "transit" would be a modelcomponent, and the
                # likelihood variable is initialized using it.
                msg = 'Expected transit to be a model component.'
                raise NotImplementedError(msg)

            for modelcomponent in self.modelcomponents:

                if 'transit' in modelcomponent:
                    mu_model = light_curve.flatten()

                if 'sincos' in modelcomponent:
                    if 'Porb' in modelcomponent:
                        k = 'orb'
                    elif 'Prot' in modelcomponent:
                        k = 'rot'

                    N_harmonics = int(modelcomponent[0])
                    for ix in range(N_harmonics):

                        sinparamnames = ['A{}{}'.format(k,ix),
                                         'omega{}'.format(k),
                                         'phi{}'.format(k)]
                        sin_params = [harmonic_d[k] for k in sinparamnames]
                        cosparamnames = ['B{}{}'.format(k,ix),
                                         'omega{}'.format(k),
                                         'phi{}'.format(k)]
                        cos_params = [harmonic_d[k] for k in cosparamnames]

                        mu_model += sin_model(sin_params, self.x_obs)
                        mu_model += cos_model(cos_params, self.x_obs)

            likelihood = pm.Normal('obs', mu=mu_model, sigma=sigma,
                                   observed=self.y_obs)

            # Get MAP estimate from model.
            map_estimate = pm.find_MAP(model=model)

            # Plot the simulated data and the maximum a posteriori model to
            # make sure that our initialization looks ok.
            y_MAP = np.zeros_like(self.x_obs)
            for modelcomponent in self.modelcomponents:
                if 'transit' in modelcomponent:
                    y_MAP += map_estimate["light_curve"].flatten()
                if 'sincos' in modelcomponent:
                    if 'Porb' in modelcomponent:
                        k = 'orb'
                    elif 'Prot' in modelcomponent:
                        k = 'rot'
                    N_harmonics = int(modelcomponent[0])
                    for ix in range(N_harmonics):
                        sinparamnames = ['A{}{}'.format(k,ix),
                                         'omega{}'.format(k),
                                         'phi{}'.format(k)]
                        sin_params = [map_estimate[k] for k in sinparamnames]
                        cosparamnames = ['B{}{}'.format(k,ix),
                                         'omega{}'.format(k),
                                         'phi{}'.format(k)]
                        cos_params = [map_estimate[k] for k in cosparamnames]
                        y_MAP += sin_model(sin_params, self.x_obs)
                        y_MAP += cos_model(cos_params, self.x_obs)

            self.y_MAP = y_MAP

            plt.figure(figsize=(14, 4))
            plt.plot(self.x_obs, self.y_obs, ".k", ms=4, label="data")
            plt.plot(self.x_obs, self.y_MAP, lw=1)
            plt.ylabel("relative flux")
            plt.xlabel("time [days]")
            _ = plt.title("map model")
            fig = plt.gcf()
            outpath = os.path.join(RESULTSDIR, 'driver_results',
                                   'test_{}_MAP.png'.format(self.modelid))
            savefig(fig, outpath, writepdf=0)

            # sample from the posterior defined by this model.
            trace = pm.sample(
                tune=self.N_samples, draws=self.N_samples, start=map_estimate,
                cores=self.N_cores, chains=self.N_chains,
                step=xo.get_dense_nuts_step(target_accept=0.9),
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate
