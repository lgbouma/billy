import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os
from copy import deepcopy
from collections import OrderedDict

import exoplanet as xo

from billy import __path__
from billy.models import sin_model, cos_model, transit_model
from billy.plotting import plot_test_data, savefig, plot_MAP_data
from billy.convenience import flatten as bflatten

RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')

LINEAR_AMPLITUDES = 1
LOG_AMPLITUDES = 0

class ModelParser:

    def __init__(self, modelid):
        self.initialize_model(modelid)

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
    Given a modelid of the form "transit_NsincosPorb_NsincosProt", and
    observed x and y values (typically time and flux), construct and
    fit the model. In other words, run the inference.

    The model implemented is of the form

    Y ~ N(
    [Mandel-Agol transit] +
    Σ_n A_n sin(n*ωt + φ) +
    Σ_n A_n cos(n*ωt + φ),
    σ^2).
    """

    def __init__(self, modelid, x_obs, y_obs, y_err, prior_d, mstar=1,
                 rstar=1, N_samples=2000, N_cores=16, N_chains=4,
                 plotdir=None, pklpath=None, overwrite=1):

        self.N_samples = N_samples
        self.N_cores = N_cores
        self.N_chains = N_chains
        self.PLOTDIR = plotdir
        self.OVERWRITE = overwrite
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.y_err = y_err
        self.mstar = mstar
        self.rstar = rstar
        self.t_exp = np.nanmedian(np.diff(x_obs))

        self.initialize_model(modelid)
        self.verify_inputdata()
        self.run_inference(prior_d, pklpath, make_threadsafe=True)


    def verify_inputdata(self):
        assert len(self.x_obs) == len(self.y_obs)
        assert isinstance(self.x_obs, np.ndarray)
        assert isinstance(self.y_obs, np.ndarray)


    def run_inference(self, prior_d, pklpath, make_threadsafe=True):

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
            _A_d, _B_d = {}, {}
            for modelcomponent in self.modelcomponents:

                if 'transit' in modelcomponent:

                    # mean = pm.Normal(
                    #     "mean", mu=prior_d['mean'], sd=0.02, testval=prior_d['mean']
                    # )
                    mean = pm.Uniform(
                        "mean", lower=prior_d['mean']-1e-2,
                        upper=prior_d['mean']+1e-2, testval=prior_d['mean']
                    )

                    t0 = pm.Normal(
                        "t0", mu=prior_d['t0'], sd=0.002, testval=prior_d['t0']
                    )

                    # logP = pm.Normal(
                    #     "logP", mu=np.log(prior_d['period']),
                    #     sd=0.001*np.abs(np.log(prior_d['period'])),
                    #     testval=np.log(prior_d['period'])
                    # )
                    # period = pm.Deterministic("period", pm.math.exp(logP))
                    period = pm.Normal(
                        'period', mu=prior_d['period'], sd=1e-3,
                        testval=prior_d['period']
                    )

                    u = xo.distributions.QuadLimbDark(
                        "u", testval=prior_d['u']
                    )

                    r = pm.Normal(
                        "r", mu=prior_d['r'], sd=0.10*prior_d['r'],
                        testval=prior_d['r']
                    )
                    # r = pm.Uniform(
                    #     "r", lower=prior_d['r']-1e-2,
                    #     upper=prior_d['r']+1e-2, testval=prior_d['r']
                    # )

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

                if 'sincos' in modelcomponent:
                    if 'Porb' in modelcomponent:
                        k = 'orb'
                    elif 'Prot' in modelcomponent:
                        k = 'rot'
                    else:
                        msg = 'expected Porb or Prot for freq specification'
                        raise NotImplementedError(msg)

                    omegakey = 'omega{}'.format(k)
                    if k == 'rot':
                        omega_d[omegakey] = pm.Normal(omegakey,
                                                      mu=prior_d[omegakey],
                                                      sd=0.01*prior_d[omegakey],
                                                      testval=prior_d[omegakey])
                        P_rot = pm.Deterministic(
                            'P_rot', pm.math.dot(1/omega_d[omegakey], 2*np.pi)
                        )

                        #omega_d[omegakey] = pm.Uniform(omegakey,
                        #                               lower=prior_d[omegakey]-1e-2,
                        #                               upper=prior_d[omegakey]+1e-2,
                        #                               testval=prior_d[omegakey])
                    elif k == 'orb':
                        # For orbital frequency, no need to declare new
                        # random variable!
                        omega_d[omegakey] = pm.Deterministic(
                            omegakey, pm.math.dot(1/period, 2*np.pi)
                        )

                    # sin and cosine terms are highly degenerate...
                    phikey = 'phi{}'.format(k)
                    if k == 'rot':
                        phi_d[phikey] = pm.Uniform(phikey,
                                                   lower=prior_d[phikey]-np.pi/8,
                                                   upper=prior_d[phikey]+np.pi/8,
                                                   testval=prior_d[phikey])
                        #phi_d[phikey] = pm.Uniform(phikey,
                        #                           lower=0,
                        #                           upper=np.pi,
                        #                           testval=prior_d[phikey])
                    elif k == 'orb':
                        # For orbital phase, no need to declare new
                        # random variable!
                        phi_d[phikey] = pm.Deterministic(
                            phikey, pm.math.dot(t0/period, 2*np.pi)
                        )

                    N_harmonics = int(modelcomponent[0])
                    for ix in range(N_harmonics):

                        if LINEAR_AMPLITUDES:
                            Akey = 'A{}{}'.format(k,ix)
                            Bkey = 'B{}{}'.format(k,ix)

                            A_d[Akey] = pm.Uniform(
                                Akey,
                                lower=-2*np.abs(prior_d[Akey]),
                                upper=2*np.abs(prior_d[Akey]),
                                testval=np.abs(prior_d[Akey]))

                            B_d[Bkey] = pm.Uniform(
                                Bkey,
                                lower=-2*np.abs(prior_d[Bkey]),
                                upper=2*np.abs(prior_d[Bkey]),
                                testval=np.abs(prior_d[Bkey]))

                        if LOG_AMPLITUDES:
                            Akey = 'A{}{}'.format(k,ix)
                            Bkey = 'B{}{}'.format(k,ix)
                            logAkey = 'logA{}{}'.format(k,ix)
                            logBkey = 'logB{}{}'.format(k,ix)

                            if k == 'rot':
                                mfact = 3
                            elif k == 'orb':
                                mfact = 10
                            _A_d[logAkey] = pm.Uniform(
                                logAkey,
                                lower=np.log(prior_d[Akey]/mfact),
                                upper=np.log(mfact*prior_d[Akey]),
                                testval=np.log(prior_d[Akey])
                            )
                            A_d[Akey] = pm.Deterministic(
                                Akey, pm.math.exp(_A_d[logAkey])
                            )

                            _B_d[logBkey] = pm.Uniform(
                                logBkey,
                                lower=np.log(prior_d[Bkey]/mfact),
                                upper=np.log(mfact*prior_d[Bkey]),
                                testval=np.log(prior_d[Bkey])
                            )
                            B_d[Bkey] = pm.Deterministic(
                                Bkey, pm.math.exp(_B_d[logBkey])
                            )

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
                    pm.Deterministic("mu_transit", light_curve.flatten())

                if 'sincos' in modelcomponent:
                    if 'Porb' in modelcomponent:
                        k = 'orb'
                    elif 'Prot' in modelcomponent:
                        k = 'rot'

                    N_harmonics = int(modelcomponent[0])
                    for ix in range(N_harmonics):

                        spnames = ['A{}{}'.format(k,ix), 'omega{}'.format(k),
                                   'phi{}'.format(k)]
                        cpnames = ['B{}{}'.format(k,ix), 'omega{}'.format(k),
                                   'phi{}'.format(k)]
                        sin_params = [harmonic_d[k] for k in spnames]
                        cos_params = [harmonic_d[k] for k in cpnames]

                        # harmonic multiplier
                        mult = ix + 1
                        sin_params[1] = pm.math.dot(sin_params[1], mult)
                        cos_params[1] = pm.math.dot(cos_params[1], mult)

                        s_mod = sin_model(sin_params, self.x_obs)
                        c_mod = cos_model(cos_params, self.x_obs)

                        mu_model += s_mod
                        mu_model += c_mod

                        # save model components (rot and orb) for plotting
                        pm.Deterministic(
                            "mu_{}sin{}".format(k,ix), s_mod
                        )
                        pm.Deterministic(
                            "mu_{}cos{}".format(k,ix), c_mod
                        )

            # track the total model to plot it
            pm.Deterministic("mu_model", mu_model)

            likelihood = pm.Normal('obs', mu=mu_model, sigma=sigma,
                                   observed=self.y_obs)

            # Get MAP estimate from model.
            map_estimate = pm.find_MAP(model=model)

            # Plot the simulated data and the maximum a posteriori model to
            # make sure that our initialization looks ok.
            self.y_MAP = map_estimate['mu_model'].flatten()

            if make_threadsafe:
                pass
            else:
                # as described in
                # https://github.com/matplotlib/matplotlib/issues/15410
                # matplotlib is not threadsafe. so do not make plots before
                # sampling, because some child processes tries to close a
                # cached file, and crashes the sampler.
                if self.PLOTDIR is None:
                    raise NotImplementedError
                outpath = os.path.join(self.PLOTDIR,
                                       'test_{}_MAP.png'.format(self.modelid))
                plot_MAP_data(self.x_obs, self.y_obs, self.y_MAP, outpath)

            # sample from the posterior defined by this model.
            trace = pm.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                step=xo.get_dense_nuts_step(target_accept=0.9),
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate
