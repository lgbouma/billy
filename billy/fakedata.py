import numpy as np, matplotlib.pyplot as plt, pandas as pd
import pickle, os, corner
from collections import OrderedDict

from billy import __path__
from billy.models import sin_model, cos_model, transit_model
from billy.plotting import plot_test_data
import billy.modelfitter as bm

RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')

class FakeDataGenerator(bm.ModelParser):

    def __init__(self, modelid, plotdir):

        self.initialize_model(modelid)
        self.make_true_params()
        self.make_fake_data()

        plot_test_data(self.x_obs, self.y_obs, self.y_mod, self.modelid,
                       outdir=plotdir)


    def make_true_params(self):
        """
        Make a dictionary with all the "true" parameters of your fake data.

        true_d = OrderedDict({'A_0': 1, 'omega_0': 2, 'phi_0': 0.3141, 'A_1':
            0.2, 'omega_1': 1.8, 'phi_1': 0.5, 'period':4.3, 't0':0.2,
            'r':0.04, 'b':0.5, 'u':[0.3,0.2], 'mean':0})
        """

        P_orb = 3*0.45
        t0_orb = 0.  # non-zero means phase offset.

        P_rot = 3*0.496
        t0_rot = 0.42

        true_d = OrderedDict()

        for modelcomponent in self.modelcomponents:

            if 'transit' in modelcomponent:
                true_d['period'] = P_orb
                true_d['t0'] = t0_orb
                true_d['r'] = 0.15
                true_d['b'] = 0.5
                true_d['u'] = [0.3,0.2]
                true_d['mean'] = 0

            if 'sincos' in modelcomponent:

                N_harmonics = int(modelcomponent[0])
                for ix in range(N_harmonics):

                    if 'Porb' in modelcomponent:
                        k = 'orb'
                    elif 'Prot' in modelcomponent:
                        k = 'rot'
                    else:
                        msg = 'expected Porb or Prot for freq specification'
                        raise NotImplementedError(msg)

                    if k == 'orb':
                        true_d['A{}{}'.format(k,ix)] = (
                            np.random.uniform(low=1e-5, high=2e-5)
                        )
                        if ix == 1:
                            true_d['B{}{}'.format(k,ix)] = (
                                np.random.uniform(low=-0.01, high=-0.009)
                            )
                        else:
                            true_d['B{}{}'.format(k,ix)] = (
                                np.random.uniform(low=1e-5, high=2e-5)
                            )
                    elif k == 'rot':
                        true_d['A{}{}'.format(k,ix)] = (
                            np.random.uniform(low=0.05, high=0.06)
                        )
                        true_d['B{}{}'.format(k,ix)] = (
                            np.random.uniform(low=1e-5, high=2e-5)
                        )

                    if 'Porb' in modelcomponent:
                        phi = 2*np.pi * t0_orb / P_orb
                    elif 'Prot' in modelcomponent:
                        phi = 2*np.pi * t0_rot / P_rot
                    true_d['phi{}'.format(k)] = phi

                    if 'Porb' in modelcomponent:
                        omega = 2*np.pi / P_orb
                    elif 'Prot' in modelcomponent:
                        omega = 2*np.pi / P_rot
                    true_d['omega{}'.format(k)] = omega

        self.true_d = true_d


    def make_fake_data(self):

        np.random.seed(42)
        t_exp = 30/(60*24)
        y_err = 2e-3
        mstar, rstar = 1, 1

        x_obs = np.arange(0, 28, t_exp)

        y_mod = np.zeros_like(x_obs)

        for modelcomponent in self.modelcomponents:

            if 'transit' in modelcomponent:
                transitparamnames = ['period', 't0', 'r', 'b', 'u', 'mean']
                transit_params = [self.true_d[k] for k in transitparamnames]

                y_mod += (
                    transit_model(transit_params, x_obs,
                                  mstar=mstar, rstar=rstar)
                )

            if 'sincos' in modelcomponent:

                N_harmonics = int(modelcomponent[0])

                for ix in range(N_harmonics):

                    if 'Porb' in modelcomponent:
                        k = 'orb'
                    elif 'Prot' in modelcomponent:
                        k = 'rot'
                    else:
                        msg = 'expected Porb or Prot for freq specification'
                        raise NotImplementedError(msg)

                    spnames = ['A{}{}'.format(k,ix), 'omega{}'.format(k),
                               'phi{}'.format(k)]
                    sin_params = [self.true_d[k] for k in spnames]

                    cpnames = ['B{}{}'.format(k,ix), 'omega{}'.format(k),
                               'phi{}'.format(k)]
                    cos_params = [self.true_d[k] for k in cpnames]

                    mult = ix + 1
                    sin_params[1] *= mult
                    cos_params[1] *= mult

                    y_mod += sin_model(sin_params, x_obs)
                    y_mod += cos_model(cos_params, x_obs)

        y_obs = y_mod + np.random.normal(scale=y_err, size=len(x_obs))

        self.x_obs = x_obs
        self.y_obs = y_obs
        self.y_mod = y_mod
        self.y_err = y_err
