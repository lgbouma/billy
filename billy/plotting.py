"""
Plots:

    plot_periodogram
    plot_test_data
    plot_MAP_data
    plot_sampleplot
    plot_splitsignal_map
    plot_phasefold_map
    plot_splitsignal_post
    plot_phasefold_post
    plot_traceplot
    plot_cornerplot

    plot_scene

    plot_hr
    plot_astrometric_excess

    plot_O_minus_C

    plot_brethren

Convenience:
    savefig
    format_ax
"""
import os, corner, pickle
from glob import glob
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from datetime import datetime
from pymc3.backends.tracetab import trace_to_dataframe
from itertools import product

from billy.convenience import flatten as bflatten
from billy.convenience import get_clean_ptfo_data
from billy.models import linear_model

from astrobase.lcmath import (
    phase_magseries, phase_bin_magseries, sigclip_magseries,
    find_lc_timegroups
)
from astrobase import periodbase

from astropy.stats import LombScargle
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.time import Time

from numpy import array as nparr

def plot_periodogram(outdir, islinear=True):

    x_obs, y_obs, y_err = get_clean_ptfo_data(binsize=None)

    period_min, period_max, N_freqs = 0.3, 0.7, int(3e3)
    frequency = np.linspace(1/period_max, 1/period_min, N_freqs)
    ls = LombScargle(x_obs, y_obs, y_err, normalization='standard')
    power = ls.power(frequency)
    period = 1/frequency

    P_rot, P_orb = 0.49914, 0.4485

    plt.close('all')
    f, ax = plt.subplots(figsize=(4,3))
    ax.plot(
        period, power, lw=0.5, c='k'
    )

    if not islinear:
        ax.set_yscale('log')

    ylim = ax.get_ylim()
    for P,c in zip([P_rot, P_orb],['C0','C1']):
        for m in [1]:
            ax.vlines(
                m*P, min(ylim), max(ylim), colors=c, alpha=0.5,
                linestyles='--', zorder=-2, linewidths=0.5
            )

    #ax.set_xlabel('Frequency [1/days]')
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('Lomb-Scargle Power')
    if not islinear:
        ax.set_ylim([1e-4, 1.2])
    ax.set_xlim([period_min, period_max])

    format_ax(ax)
    outpath = os.path.join(outdir, 'periodogram.png')
    savefig(f, outpath)


def plot_test_data(x_obs, y_obs, y_mod, modelid, outdir):
    plt.close('all')
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111, xlabel='x_obs', ylabel='y_obs',
                         title='Generated data and underlying model')
    ax.plot(x_obs, y_obs, 'x', label='sampled data')
    ax.plot(x_obs, y_mod, label='true regression line', lw=2.)
    plt.legend(loc=0)
    outpath = os.path.join(outdir, 'test_{}_data.png'.format(modelid))
    format_ax(ax)
    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_MAP_data(x_obs, y_obs, y_MAP, outpath):
    plt.close('all')
    plt.figure(figsize=(14, 4))
    plt.plot(x_obs, y_obs, ".k", ms=4, label="data")
    plt.plot(x_obs, y_MAP, lw=1)
    plt.ylabel("relative flux")
    plt.xlabel("time [days]")
    _ = plt.title("MAP model")
    fig = plt.gcf()
    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_sampleplot(m, outpath, N_samples=100):

    if os.path.exists(outpath) and not m.OVERWRITE:
        return

    plt.close('all')
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(m.x_obs, m.y_obs, ".k", ms=4, label="data", zorder=N_samples+1)
    ax.plot(m.x_obs, m.map_estimate['mu_model'], lw=0.5, label='MAP',
            zorder=N_samples+2, color='C1', alpha=1)

    np.random.seed(42)
    y_mod_samples = (
        m.trace.mu_model[
            np.random.choice(
                m.trace.mu_model.shape[0], N_samples, replace=False
            ), :
        ]
    )

    for i in range(N_samples):
        if i % 10 == 0:
            print(i)
        ax.plot(m.x_obs, y_mod_samples[i,:], color='C0', alpha=0.3,
                rasterized=True, lw=0.5)

    ax.set_ylabel("relative flux")
    ax.set_xlabel("time [days]")
    ax.legend(loc='best')
    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_splitsignal_map(m, outpath, part='i'):
    """
    y_obs + y_MAP + y_rot + y_orb
    things at rotation frequency
    things at orbital frequency
    """
    ngroups, groupinds = find_lc_timegroups(m.x_obs, mingap=1.0)
    assert ngroups == 2
    if part == 'i':
        g = groupinds[0]
    elif part == 'ii':
        g = groupinds[1]
    else:
        raise NotImplementedError

    plt.close('all')
    # 8.5x11 is letter paper. x10 allows space for caption.
    fig, axs = plt.subplots(nrows=4, figsize=(8.5, 10), sharex=True)

    axs[0].set_ylabel('Raw flux', fontsize='x-large')
    # axs[0].set_ylabel('f', fontsize='x-large')
    axs[0].plot(m.x_obs[g], m.y_obs[g], ".k", ms=4, label="data", zorder=2,
                rasterized=True)
    axs[0].plot(m.x_obs[g], m.map_estimate['mu_model'][g], lw=0.5, label='MAP',
                color='C0', alpha=1, zorder=1)

    y_tra = m.map_estimate['mu_transit']
    for ix, f in enumerate(['rot', 'orb']):
        N_harmonics = int([c for c in m.modelcomponents if f in c][0][0])
        yval = np.zeros_like(m.x_obs)
        for n in range(N_harmonics):
            k0 = "mu_{}sin{}".format(f,n)
            k1 = "mu_{}cos{}".format(f,n)
            yval += m.map_estimate[k0]
            yval += m.map_estimate[k1]
        if f == 'rot':
            y_rot = yval
            # axs[0].plot(m.x_obs, y_rot, lw=0.5, label='model '+f,
            #             color='C{}'.format(ix+1), alpha=1, zorder=ix+3)
        if f == 'orb':
            y_orb = yval + y_tra
            # axs[0].plot(m.x_obs, y_orb, lw=0.5, label='model '+f,
            #             color='C{}'.format(ix+1), alpha=1, zorder=ix+3)

    axs[1].set_ylabel('Longer period',
                      fontsize='x-large')
    # axs[1].set_ylabel('$f_{{\mathrm{{\ell}}}} = f - f_{{\mathrm{{s}}}}$',
    #                   fontsize='x-large')
    axs[1].plot(m.x_obs[g], m.y_obs[g]-y_orb[g], ".k", ms=4, label="data-orb",
                zorder=2, rasterized=True)
    axs[1].plot(m.x_obs[g], m.map_estimate['mu_model'][g]-y_orb[g], lw=0.5,
                label='model-orb', color='C0', alpha=1, zorder=1)

    axs[2].set_ylabel('Shorter period',
                      fontsize='x-large')
    # axs[2].set_ylabel('$f_{{\mathrm{{s}}}} = f - f_{{\mathrm{{\ell}}}}$',
    #                   fontsize='x-large')
    axs[2].plot(m.x_obs[g], m.y_obs[g]-y_rot[g], ".k", ms=4, label="data-rot",
                zorder=2, rasterized=True)
    axs[2].plot(m.x_obs[g], m.map_estimate['mu_model'][g]-y_rot[g], lw=0.5,
                label='model-rot', color='C0', alpha=1, zorder=1)

    axs[3].set_ylabel('Residual',
                      fontsize='x-large')
    # axs[3].set_ylabel('$f - f_{{\mathrm{{s}}}} - f_{{\mathrm{{\ell}}}}$',
    #                   fontsize='x-large')
    axs[3].plot(m.x_obs[g], m.y_obs[g]-m.map_estimate['mu_model'][g], ".k",
                ms=4, label="data", zorder=2, rasterized=True)
    axs[3].plot(m.x_obs[g],
                m.map_estimate['mu_model'][g]-m.map_estimate['mu_model'][g],
                lw=0.5, label='model', color='C0', alpha=1, zorder=1)


    axs[-1].set_xlabel("Time [days]", fontsize='x-large')
    for a in axs:
        #a.legend()
        format_ax(a)
        a.set_ylim((-.075, .075))
        if part == 'i':
            a.set_xlim((0, 9))
        else:
            a.set_xlim((10, 20.3))

    props = dict(boxstyle='square', facecolor='white', alpha=0.9, pad=0.15,
                 linewidth=0)
    if part == 'i':
        axs[3].text(0.97, 0.03, 'Orbit 19', ha='right', va='bottom',
                    transform=axs[3].transAxes, bbox=props, zorder=3,
                    fontsize='x-large')
    else:
        axs[3].text(0.97, 0.03, 'Orbit 20', ha='right', va='bottom',
                    transform=axs[3].transAxes, bbox=props, zorder=3,
                    fontsize='x-large')

    fig.tight_layout(h_pad=0., w_pad=0.)

    if not os.path.exists(outpath) or m.OVERWRITE:
        savefig(fig, outpath, writepdf=1, dpi=300)

    ydict = {
        'x_obs': m.x_obs,
        'y_obs': m.y_obs,
        'y_orb': m.y_obs-y_rot,
        'y_rot': m.y_obs-y_orb,
        'y_resid': m.y_obs-m.map_estimate['mu_model'],
        'y_mod_tra': y_tra,
        'y_mod_rot': y_orb,
        'y_mod_orb': y_rot,
        'y_mod': m.map_estimate['mu_model'],
        'y_err': m.y_err
    }
    return ydict


def plot_splitsignal_map_periodogram(ydict, outpath):
    """
    y_obs + y_MAP + y_rot + y_orb
    things at rotation frequency
    things at orbital frequency
    """

    P_rot, P_orb = 0.49914, 0.4485

    period_min, period_max, N_freqs = 0.3, 0.7, int(3e3)
    frequency = np.linspace(1/period_max, 1/period_min, N_freqs)
    period = 1/frequency

    ytypes = ['y_obs', 'y_rot', 'y_orb', 'y_resid']
    ylabels = ['power (raw)', 'power (rot)', 'power (orb)', 'power (resid)']

    ls_d = {}
    for k in ytypes:
        ls = LombScargle(ydict['x_obs'], ydict[k], ydict['y_err'],
                         normalization='standard')
        if not k == 'y_resid':
            power = ls.power(frequency)
        else:
            _period_min, _period_max, _N_freqs = 1, 20, int(2e4)
            _frequency = np.linspace(1/_period_max, 1/_period_min, _N_freqs)
            _period = 1/_frequency
            power = ls.power(_frequency)

        ls_d[k] = power
        ls_fap = ls.false_alarm_probability(power.max())
        msg = '{}: FAP = {:.2e}'.format(k, ls_fap)
        print(msg)

    plt.close('all')
    fig, axs = plt.subplots(nrows=4, figsize=(4, 12), sharex=False)

    for ax, k, l in zip(axs, ytypes, ylabels):

        if not k == 'y_resid':
            ax.plot(period, ls_d[k], lw=0.5, c='k')
        else:
            ax.plot(_period, ls_d[k], lw=0.5, c='k')

        ylim = ax.get_ylim()
        for P,c in zip([P_rot, P_orb],['C0','C1']):
            for m in [1]:
                ax.vlines(
                    m*P, min(ylim), max(ylim), colors=c, alpha=0.5,
                    linestyles='--', zorder=-2, linewidths=0.5
                )
        ax.set_ylim(ylim)
        ax.set_ylabel(l)
        if not k == 'y_resid':
            ax.set_xlim([period_min, period_max])
        else:
            ax.set_xlim([_period_min, _period_max])
            ax.set_xscale('log')

    axs[-1].set_xlabel('Period [days]')

    for a in axs:
        # a.legend()
        format_ax(a)

    fig.tight_layout()
    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_phasefold_map(m, d, outpath):

    if os.path.exists(outpath) and not m.OVERWRITE:
        return

    # recover periods and epochs.
    P_rot = 2*np.pi/float(m.map_estimate['omegarot'])
    t0_rot = float(m.map_estimate['phirot']) * P_rot / (2*np.pi)
    P_orb = float(m.map_estimate['period'])
    t0_orb = float(m.map_estimate['t0'])

    # phase and bin them.
    orb_d = phase_magseries(
        d['x_obs'], d['y_orb'], P_orb, t0_orb, wrap=True, sort=True
    )
    orb_bd = phase_bin_magseries(
        orb_d['phase'], orb_d['mags'], binsize=0.01
    )
    rot_d = phase_magseries(
        d['x_obs'], d['y_rot'], P_rot, t0_rot, wrap=True, sort=True
    )
    rot_bd = phase_bin_magseries(
        rot_d['phase'], rot_d['mags'], binsize=0.01
    )

    # make tha plot
    plt.close('all')
    fig, axs = plt.subplots(nrows=2, figsize=(6, 6), sharex=True)

    axs[0].scatter(rot_d['phase'], rot_d['mags'], color='gray', s=2, alpha=0.8,
                   zorder=4, linewidths=0, rasterized=True)
    axs[0].scatter(rot_bd['binnedphases'], rot_bd['binnedmags'], color='black',
                   s=8, alpha=1, zorder=5, linewidths=0)

    txt0 = '$P_{{\mathrm{{\ell}}}}$ = {:.5f}$\,$d'.format(P_rot)
    props = dict(boxstyle='square', facecolor='white', alpha=0.9, pad=0.15,
                 linewidth=0)

    axs[0].text(0.98, 0.98, txt0, ha='right', va='top',
                transform=axs[0].transAxes, bbox=props, zorder=3)
    #axs[0].set_ylabel('$f_{{\mathrm{{\ell}}}} = f - f_{{\mathrm{{s}}}}$',
    #                  fontsize='large')
    axs[0].set_ylabel('Longer period', fontsize='large')

    axs[1].scatter(orb_d['phase'], orb_d['mags'], color='gray', s=2, alpha=0.8,
                   zorder=4, linewidths=0, rasterized=True)
    axs[1].scatter(orb_bd['binnedphases'], orb_bd['binnedmags'], color='black',
                   s=8, alpha=1, zorder=5, linewidths=0)

    out_d = {
        'orb_d': orb_d,
        'orb_bd': orb_bd,
        'rot_d': rot_d,
        'rot_bd': rot_bd
    }
    pklpath = os.path.join(
        os.path.dirname(outpath),
        os.path.basename(outpath).replace('.png','_points.pkl')
    )
    with open(pklpath, 'wb') as buff:
        pickle.dump(out_d, buff)
    print('made {}'.format(pklpath))

    txt1 = '$P_{{\mathrm{{s}}}}$ = {:.5f}$\,$d'.format(P_orb)
    axs[1].text(0.98, 0.98, txt1, ha='right', va='top',
                transform=axs[1].transAxes, bbox=props, zorder=3)

    axs[1].set_ylabel('Shorter period',
                      fontsize='large')
    #axs[1].set_ylabel('$f_{{\mathrm{{s}}}} = f - f_{{\mathrm{{\ell}}}}$',
    #                  fontsize='large')

    axs[1].set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    axs[1].set_yticks([-0.04, -0.02, 0, 0.02, 0.04])

    axs[-1].set_xlabel('Phase', fontsize='large')

    for a in axs:
        a.grid(which='major', axis='both', linestyle='--', zorder=-3,
               alpha=0.5, color='gray', linewidth=0.5)

    # pct_80 = np.percentile(results.model_folded_model, 80)
    # pct_20 = np.percentile(results.model_folded_model, 20)
    # center = np.nanmedian(results.model_folded_model)
    # delta_y = (10/6)*np.abs(pct_80 - pct_20)
    # plt.ylim(( center-0.7*delta_y, center+0.7*delta_y ))

    for a in axs:
        a.set_xlim((-1, 1))
        format_ax(a)
    axs[0].set_ylim((-0.075, 0.075))
    axs[1].set_ylim((-0.045, 0.045))
    fig.tight_layout()
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_splitsignal_post(m, outpath):
    """
    y_obs + y_mod + y_rot + y_orb
    things at rotation frequency
    things at orbital frequency
    """

    # get y_mod, y_rot, y_orb, y_tra. here: cheat. just randomly select 1 from
    # posterior (TODO: take the median parameters, +generate the model instead)
    np.random.seed(42)
    sel = np.random.choice(m.trace.mu_model.shape[0], 1)
    y_mod = m.trace.mu_model[sel, :].flatten()
    y_tra = m.trace.mu_transit[sel, :].flatten()

    y_orb, y_rot = np.zeros_like(m.x_obs), np.zeros_like(m.x_obs)
    for modelcomponent in m.modelcomponents:
        if 'rot' in modelcomponent:
            N_harmonics = int(modelcomponent[0])
            for ix in range(N_harmonics):
                y_rot += m.trace['mu_rotsin{}'.format(ix)][sel, :].flatten()
                y_rot += m.trace['mu_rotcos{}'.format(ix)][sel, :].flatten()

        if 'orb' in modelcomponent:
            N_harmonics = int(modelcomponent[0])
            for ix in range(N_harmonics):
                y_orb += m.trace['mu_orbsin{}'.format(ix)][sel, :].flatten()
                y_orb += m.trace['mu_orbcos{}'.format(ix)][sel, :].flatten()

    # make the plot!
    plt.close('all')
    fig, axs = plt.subplots(nrows=4, figsize=(14, 12), sharex=True)

    axs[0].set_ylabel('flux')
    axs[0].plot(m.x_obs, m.y_obs, ".k", ms=4, label="data")
    axs[0].plot(m.x_obs, y_mod, lw=0.5, label='model',
                color='C0', alpha=1, zorder=5)

    for ix, f in enumerate(['rot', 'orb']):
        if f == 'rot':
            axs[0].plot(m.x_obs, y_rot, lw=0.5, label='model '+f,
                        color='C{}'.format(ix+1), alpha=1, zorder=ix+3)
        if f == 'orb':
            axs[0].plot(m.x_obs, y_orb+y_tra, lw=0.5, label='model '+f,
                        color='C{}'.format(ix+1), alpha=1, zorder=ix+3)

    axs[1].set_ylabel('flux-orb (rot)')
    axs[1].plot(m.x_obs, m.y_obs-y_orb-y_tra, ".k", ms=4, label="data-orb")
    axs[1].plot(m.x_obs, y_mod-y_orb-y_tra, lw=0.5,
                label='model-orb', color='C0', alpha=1, zorder=5)

    axs[2].set_ylabel('flux-rot (orb)')
    axs[2].plot(m.x_obs, m.y_obs-y_rot, ".k", ms=4, label="data-rot")
    axs[2].plot(m.x_obs, y_mod-y_rot, lw=0.5,
                label='model-rot', color='C0', alpha=1, zorder=5)

    axs[3].set_ylabel('flux-model')
    axs[3].plot(m.x_obs, m.y_obs-y_mod, ".k", ms=4, label="data")
    axs[3].plot(m.x_obs, y_mod-y_mod, lw=0.5, label='model',
                color='C0', alpha=1, zorder=5)

    axs[-1].set_xlabel("time [days]")
    for a in axs:
        a.legend()
        format_ax(a)
    fig.tight_layout()
    savefig(fig, outpath, writepdf=0, dpi=300)

    ydict = {
        'x_obs': m.x_obs,
        'y_obs': m.y_obs,
        'y_orb': m.y_obs-y_rot,
        'y_rot': m.y_obs-y_orb,
        'y_mod_tra': y_tra,
        'y_mod_rot': y_orb,
        'y_mod_orb': y_rot
    }
    return ydict


def plot_phasefold_post(m, d, outpath):

    # recover periods and epochs.
    P_rot = 2*np.pi/float(np.nanmedian(m.trace['omegarot']))
    t0_rot = float(np.nanmedian(m.trace['phirot'])) * P_rot / (2*np.pi)
    P_orb = float(np.nanmedian(m.trace['period']))
    t0_orb = float(np.nanmedian(m.trace['t0']))

    # phase and bin them.
    orb_d = phase_magseries(
        d['x_obs'], d['y_orb'], P_orb, t0_orb, wrap=True, sort=True
    )
    orb_bd = phase_bin_magseries(
        orb_d['phase'], orb_d['mags'], binsize=0.01
    )
    rot_d = phase_magseries(
        d['x_obs'], d['y_rot'], P_rot, t0_rot, wrap=True, sort=True
    )
    rot_bd = phase_bin_magseries(
        rot_d['phase'], rot_d['mags'], binsize=0.01
    )

    # make tha plot
    plt.close('all')
    fig, axs = plt.subplots(nrows=2, figsize=(6, 8), sharex=True)

    axs[0].scatter(rot_d['phase'], rot_d['mags'], color='gray', s=2, alpha=0.8,
                   zorder=4, linewidths=0)
    axs[0].scatter(rot_bd['binnedphases'], rot_bd['binnedmags'], color='black',
                   s=8, alpha=1, zorder=5, linewidths=0)
    txt0 = 'Prot {:.5f}d, t0 {:.5f}'.format(P_rot, t0_rot)
    axs[0].text(0.98, 0.98, txt0, ha='right', va='top',
                transform=axs[0].transAxes)
    axs[0].set_ylabel('flux-orb (rot)')

    axs[1].scatter(orb_d['phase'], orb_d['mags'], color='gray', s=2, alpha=0.8,
                   zorder=4, linewidths=0)
    axs[1].scatter(orb_bd['binnedphases'], orb_bd['binnedmags'], color='black',
                   s=8, alpha=1, zorder=5, linewidths=0)
    txt1 = 'Porb {:.5f}d, t0 {:.5f}'.format(P_orb, t0_orb)
    axs[1].text(0.98, 0.98, txt1, ha='right', va='top',
                transform=axs[1].transAxes)
    axs[1].set_ylabel('flux-rot (orb)')
    axs[1].set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])

    axs[-1].set_xlabel('phase')

    for a in axs:
        a.grid(which='major', axis='both', linestyle='--', zorder=-3,
                 alpha=0.5, color='gray')

    # pct_80 = np.percentile(results.model_folded_model, 80)
    # pct_20 = np.percentile(results.model_folded_model, 20)
    # center = np.nanmedian(results.model_folded_model)
    # delta_y = (10/6)*np.abs(pct_80 - pct_20)
    # plt.ylim(( center-0.7*delta_y, center+0.7*delta_y ))

    for a in axs:
        a.set_xlim((-0.1-0.5, 1.1-0.5))
        format_ax(a)
    fig.tight_layout()
    savefig(fig, outpath, writepdf=0, dpi=300)



def plot_traceplot(m, outpath):
    # trace plot from PyMC3
    if not os.path.exists(outpath):
        plt.figure(figsize=(7, 7))
        pm.traceplot(m.trace[100:])
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close('all')


def plot_cornerplot(true_d, m, outpath):

    if os.path.exists(outpath) and not m.OVERWRITE:
        return

    # corner plot of posterior samples
    plt.close('all')
    trace_df = trace_to_dataframe(m.trace, varnames=list(true_d.keys()))
    truths = [true_d[k] for k in true_d.keys()]
    truths = list(bflatten(truths))
    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        truths=truths, title_fmt='.2g')
    savefig(fig, outpath, writepdf=0, dpi=100)


def savefig(fig, figpath, writepdf=True, dpi=450):
    fig.savefig(figpath, dpi=dpi, bbox_inches='tight')
    print('{}: made {}'.format(datetime.utcnow().isoformat(), figpath))

    if writepdf:
        pdffigpath = figpath.replace('.png','.pdf')
        fig.savefig(pdffigpath, bbox_inches='tight', rasterized=True, dpi=dpi)
        print('{}: made {}'.format(datetime.utcnow().isoformat(), pdffigpath))

    plt.close('all')


def format_ax(ax):
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')


def plot_scene(c_obj, img_wcs, img, outpath, Tmag_cutoff=17, showcolorbar=0,
               ap_mask=0, bkgd_mask=0):

    from astrobase.plotbase import skyview_stamp
    from astropy.wcs import WCS
    from astroquery.mast import Catalogs
    import astropy.visualization as vis
    import matplotlib as mpl
    from matplotlib import patches

    plt.close('all')

    # standard tick formatting fails for these images.
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    #
    # wcs information parsing
    # follow Clara Brasseur's https://github.com/ceb8/tessworkshop_wcs_hack
    # (this is from the CDIPS vetting reports...)
    #
    radius = 5.0*u.arcminute

    nbhr_stars = Catalogs.query_region(
        "{} {}".format(float(c_obj.ra.value), float(c_obj.dec.value)),
        catalog="TIC",
        radius=radius
    )

    try:
        px,py = img_wcs.all_world2pix(
            nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['ra'],
            nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['dec'],
            0
        )
    except Exception as e:
        print('ERR! wcs all_world2pix got {}'.format(repr(e)))
        raise(e)

    ticids = nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['ID']
    tmags = nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['Tmag']

    sel = (px > 0) & (px < 10) & (py > 0) & (py < 10)
    px,py = px[sel], py[sel]
    ticids, tmags = ticids[sel], tmags[sel]

    ra, dec = float(c_obj.ra.value), float(c_obj.dec.value)
    target_x, target_y = img_wcs.all_world2pix(ra,dec,0)

    # geometry: there are TWO coordinate axes. (x,y) and (ra,dec). To get their
    # relative orientations, the WCS and ignoring curvature will usually work.
    shiftra_x, shiftra_y = img_wcs.all_world2pix(ra+1e-4,dec,0)
    shiftdec_x, shiftdec_y = img_wcs.all_world2pix(ra,dec+1e-4,0)

    ###########
    # get DSS #
    ###########
    ra = c_obj.ra.value
    dec = c_obj.dec.value
    sizepix = 220
    try:
        dss, dss_hdr = skyview_stamp(ra, dec, survey='DSS2 Red',
                                     scaling='Linear', convolvewith=None,
                                     sizepix=sizepix, flip=False,
                                     cachedir='~/.astrobase/stamp-cache',
                                     verbose=True, savewcsheader=True)
    except (OSError, IndexError, TypeError) as e:
        print('downloaded FITS appears to be corrupt, retrying...')
        try:
            dss, dss_hdr = skyview_stamp(ra, dec, survey='DSS2 Red',
                                         scaling='Linear', convolvewith=None,
                                         sizepix=sizepix, flip=False,
                                         cachedir='~/.astrobase/stamp-cache',
                                         verbose=True, savewcsheader=True,
                                         forcefetch=True)

        except Exception as e:
            print('failed to get DSS stamp ra {} dec {}, error was {}'.
                  format(ra, dec, repr(e)))
            return None, None


    ##########################################

    plt.close('all')
    fig = plt.figure(figsize=(4,9))

    # ax0: TESS
    # ax1: DSS
    ax0 = plt.subplot2grid((2, 1), (0, 0), projection=img_wcs)
    ax1 = plt.subplot2grid((2, 1), (1, 0), projection=WCS(dss_hdr))

    ##########################################

    #
    # ax0: img
    #

    #interval = vis.PercentileInterval(99.99)
    interval = vis.AsymmetricPercentileInterval(20,99)
    vmin,vmax = interval.get_limits(img)
    norm = vis.ImageNormalize(
        vmin=vmin, vmax=vmax, stretch=vis.LogStretch(1000))

    cset0 = ax0.imshow(img, cmap=plt.cm.gray_r, origin='lower', zorder=1,
                       norm=norm)

    if isinstance(ap_mask, np.ndarray):
        for x,y in product(range(10),range(10)):
            if ap_mask[y,x]:
                ax0.add_patch(
                    patches.Rectangle(
                        (x-.5, y-.5), 1, 1, hatch='//', fill=False, snap=False,
                        linewidth=0., zorder=2, alpha=0.7, rasterized=True
                    )
                )

    if isinstance(bkgd_mask, np.ndarray):
        for x,y in product(range(10),range(10)):
            if bkgd_mask[y,x]:
                ax0.add_patch(
                    patches.Rectangle(
                        (x-.5, y-.5), 1, 1, hatch='x', fill=False, snap=False,
                        linewidth=0., zorder=2, alpha=0.7, rasterized=True
                    )
                )

    ax0.scatter(px, py, marker='x', c='C1', s=20, rasterized=True, zorder=3,
                linewidths=0.8)
    ax0.plot(target_x, target_y, mew=0.5, zorder=5, markerfacecolor='yellow',
             markersize=15, marker='*', color='k', lw=0)

    ax0.text(3.2, 5, 'A', fontsize=16, color='C1', zorder=6, style='italic')

    ax0.set_title('TESS', fontsize='xx-large')

    if showcolorbar:
        cb0 = fig.colorbar(cset0, ax=ax0, extend='neither', fraction=0.046, pad=0.04)

    #
    # ax1: DSS
    #
    cset1 = ax1.imshow(dss, origin='lower', cmap=plt.cm.gray_r)

    ax1.grid(ls='--', alpha=0.5)
    ax1.set_title('DSS2 Red', fontsize='xx-large')
    if showcolorbar:
        cb1 = fig.colorbar(cset1, ax=ax1, extend='neither', fraction=0.046,
                           pad=0.04)

    # DSS is ~1 arcsecond per pixel. overplot apertures on axes 6,7
    for ix, radius_px in enumerate([21,21*1.5,21*2.25]):
        circle = plt.Circle((sizepix/2, sizepix/2), radius_px,
                            color='C{}'.format(ix), fill=False, zorder=5+ix)
        ax1.add_artist(circle)

    #
    # ITNERMEDIATE SINCE TESS IMAGES NOW PLOTTED
    #
    for ax in [ax0]:
        ax.grid(ls='--', alpha=0.5)
        if shiftra_x - target_x > 0:
            # want RA to increase to the left (almost E)
            ax.invert_xaxis()
        if shiftdec_y - target_y < 0:
            # want DEC to increase up (almost N)
            ax.invert_yaxis()

    for ax in [ax0,ax1]:
        format_ax(ax)
        ax.set_xlabel(r'$\alpha_{2000}$')
        ax.set_ylabel(r'$\delta_{2000}$')

    if showcolorbar:
        fig.tight_layout(h_pad=-8, w_pad=-8)
    else:
        fig.tight_layout(h_pad=1, w_pad=1)

    savefig(fig, outpath, dpi=300)


def plot_hr(outdir):

    pklpath = '/Users/luke/Dropbox/proj/billy/results/cluster_membership/nbhd_info_3222255959210123904.pkl'
    info = pickle.load(open(pklpath, 'rb'))
    (targetname, groupname, group_df_dr2, target_df, nbhd_df,
     cutoff_probability, pmdec_min, pmdec_max, pmra_min, pmra_max,
     group_in_k13, group_in_cg18, group_in_kc19, group_in_k18
    ) = info

    ##########

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    nbhd_yval = np.array([nbhd_df['phot_g_mean_mag'] +
                          5*np.log10(nbhd_df['parallax']/1e3) + 5])
    ax.scatter(
        nbhd_df['phot_bp_mean_mag']-nbhd_df['phot_rp_mean_mag'], nbhd_yval,
        c='gray', alpha=1., zorder=2, s=7, rasterized=True, linewidths=0,
        label='Neighborhood', marker='.'
    )

    yval = group_df_dr2['phot_g_mean_mag'] + 5*np.log10(group_df_dr2['parallax']/1e3) + 5
    ax.scatter(
        group_df_dr2['phot_bp_mean_mag']-group_df_dr2['phot_rp_mean_mag'],
        yval,
        c='k', alpha=1., zorder=3, s=9, rasterized=True, linewidths=0,
        label='K+18 members'
    )

    target_yval = np.array([target_df['phot_g_mean_mag'] +
                            5*np.log10(target_df['parallax']/1e3) + 5])
    ax.plot(
        target_df['phot_bp_mean_mag']-target_df['phot_rp_mean_mag'],
        target_yval,
        alpha=1, mew=0.5, zorder=8, label='PTFO 8-8695', markerfacecolor='yellow',
        markersize=12, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', handletextpad=0.1, fontsize='small')
    ax.set_ylabel('G + $5\log_{10}(\omega_{\mathrm{as}}) + 5$', fontsize='large')
    ax.set_xlabel('Bp - Rp', fontsize='large')

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))

    ax.set_xlim((0.9, 3.1))
    ax.set_ylim((9.5, 4.5))

    # # set M_omega y limits
    # min_y = np.nanmin(np.array([np.nanpercentile(nbhd_yval, 2), target_yval]))
    # max_y = np.nanmax(np.array([np.nanpercentile(nbhd_yval, 98), target_yval]))
    # edge_y = 0.01*(max_y - min_y)
    # momega_ylim = [max_y+edge_y, min_y-edge_y]
    # ax.set_ylim(momega_ylim)

    format_ax(ax)
    outpath = os.path.join(outdir, 'hr.png')
    savefig(f, outpath)


def plot_astrometric_excess(outdir):

    varamppath = '/Users/luke/Dropbox/proj/billy/data/25ori-1/var_amps.csv'
    va_df = pd.read_csv(varamppath)

    pklpath = '/Users/luke/Dropbox/proj/billy/results/cluster_membership/nbhd_info_3222255959210123904.pkl'
    d = pickle.load(open(pklpath, 'rb'))
    g_df = d[2]

    g_df = g_df.merge(va_df, on='source_id', how='left')

    chisq_red = g_df.astrometric_chi2_al / (g_df.astrometric_n_obs_al - 5)

    ######

    f,ax = plt.subplots(figsize=(4,3))

    ax.scatter(g_df.phot_rp_mean_mag, chisq_red, c='gray', zorder=3, s=3,
               marker='s', rasterized=True, linewidths=0, label='All members',
               alpha=1)

    ptfosel = (g_df.source_id.astype(str) == '3222255959210123904')
    ptfo_amp = float(g_df[ptfosel].var_amp)
    sel = (g_df.var_amp > ptfo_amp)

    ax.scatter(g_df[sel].phot_rp_mean_mag, chisq_red[sel], c='k', alpha=1.,
               zorder=4, s=12,
               rasterized=True, linewidths=0, marker='s',
               label='Amplitude > {:.1f}%'.format(ptfo_amp*100))

    ax.plot(
        g_df[ptfosel].phot_rp_mean_mag,
        chisq_red[ptfosel],
        alpha=1, mew=0.5, zorder=8, label='PTFO 8-8695', markerfacecolor='yellow',
        markersize=12, marker='*', color='black', lw=0
    )

    ax.set_xlabel('Rp', fontsize='large')
    ax.set_ylabel('Astrometric Reduced $\chi^2$', fontsize='large')
    ax.set_xlim([7.5,16.5])
    ax.set_ylim([0.3,8.7])

    ax.legend(loc='best', fontsize='small')

    format_ax(ax)
    outpath = os.path.join(outdir, 'astrometric_excess.png')
    savefig(f, outpath)


def plot_O_minus_C(
    x, y, sigma_y, theta_linear, refs, savpath=None, xlabel='Epoch',
    ylabel='Deviation from constant period [min]', xlim=None, ylim=None,
    ylim1=None, include_all_points=False, onlytransits=False):

    xfit = np.linspace(10*np.min(x), 10*np.max(x), 10000)

    if not onlytransits:
        raise NotImplementedError

    fig, a0 = plt.subplots(nrows=1, ncols=1, figsize=(4*1.3,3*1.3))

    refs = refs.astype(str)
    istess = ( np.core.defchararray.find(refs, 'tess')!=-1 )

    O_m_C = (y - linear_model(theta_linear, x)) / theta_linear[1]

    a0.errorbar(x, O_m_C, sigma_y, fmt='.k', ecolor='black', zorder=10, mew=0,
                ms=5, elinewidth=1, alpha=1)

    a0.plot(x[istess], O_m_C[istess], alpha=1, mew=0.5,
            zorder=8, label='binned TESS', markerfacecolor='yellow',
            markersize=15, marker='*', color='black', lw=0)

    P_short = 0.44854
    P_long = 0.49914
    phase_diff = np.abs(P_long - P_short) / P_short
    a0.hlines(
        phase_diff, min(xlim), max(xlim), colors='gray', alpha=0.5,
        linestyles='--', zorder=-2, linewidths=0.5
    )
    a0.hlines(
        -phase_diff, min(xlim), max(xlim), colors='gray', alpha=0.5,
        linestyles='--', zorder=-2, linewidths=0.5
    )
    a0.hlines(
        0, min(xlim), max(xlim), colors='gray', alpha=0.5,
        linestyles='-', zorder=-2, linewidths=0.5
    )


    props = dict(boxstyle='square', facecolor='white', alpha=0.9, pad=0.15,
                 linewidth=0)
    txt = (
        '$t_{{\mathrm{{s}}}}$ = {:.6f}\n$P_{{\mathrm{{s}}}}$ = {:.6f}$\,$d'.
        format(theta_linear[0], theta_linear[1])
    )
    a0.text(0.03, 0.03, txt, ha='left', va='bottom', transform=a0.transAxes,
            bbox=props, zorder=3)


    # # transit axis
    # for e, tm, err in zip(x[~istess],y[~istess],sigma_y[~istess]):
    #     a0.errorbar(e,
    #                 nparr(tm-linear_model(theta_linear, e)),
    #                 err,
    #                 fmt='.k', ecolor='black', zorder=10, mew=0,
    #                 ms=7,
    #                 elinewidth=1,
    #                 alpha= 1-(err/np.max(sigma_y))**(1/2) + 0.1
    #                )

    # # for legend
    # a0.errorbar(9001, 9001, np.mean(err),
    #             fmt='.k', ecolor='black', zorder=9, alpha=1, mew=1, ms=3,
    #             elinewidth=1, label='pre-TESS')


    # # bin TESS points
    # tess_x = x[istess]
    # tess_y = y[istess]
    # tess_sigma_y = sigma_y[istess]

    # bin_tess_y = np.average(nparr(tess_y-linear_model(theta_linear, tess_x)),
    #                         weights=1/tess_sigma_y**2)
    # bin_tess_sigma_y = np.mean(tess_sigma_y)/len(tess_y)**(1/2)
    # bin_tess_x = np.median(tess_x)

    # print('\n----- error on binned tess measurement -----\n')
    # print('{:.2f} seconds'.format(bin_tess_sigma_y*60))

    # a0.plot(bin_tess_x, bin_tess_y, alpha=1, mew=0.5,
    #         zorder=42, label='binned TESS', markerfacecolor='yellow',
    #         markersize=9, marker='*', color='black', lw=0)
    # a0.errorbar(bin_tess_x, bin_tess_y, bin_tess_sigma_y,
    #             alpha=1, zorder=11, label='binned TESS',
    #             fmt='s', mfc='firebrick', elinewidth=1,
    #             ms=0,
    #             mec='black',mew=1,
    #             ecolor='black')


    if include_all_points:
        raise NotImplementedError

    # add "time" axis on top
    # make twin axis to show year on top

    period = theta_linear[1]*u.day
    t0 = theta_linear[0]*u.day
    transittimes = np.linspace(xlim[0], xlim[1], 100)*period + t0
    times = Time(transittimes, format='jd', scale='tdb')
    a_top = a0.twiny()
    a_top.scatter(times.decimalyear, np.zeros_like(times), s=0)
    a_top.set_xlabel('Year', fontsize='large')

    # hidden point for a1 legend
    #a1.plot(1500, 3, alpha=1, mew=0.5,
    #        zorder=-3, label='binned TESS time', markerfacecolor='yellow',
    #        markersize=9, marker='*', color='black', lw=0)

    #if not include_all_points:
    #    a0.legend(loc='upper right', fontsize='x-small', framealpha=1)
    #else:
    #    a0.legend(loc='upper right', fontsize='x-small', framealpha=1)

    # for a in [a0,a_top]:
    #     format_ax(a)

    a0.set_xlim(xlim)
    # a0.set_ylim((-1.6, 1.6))

    a0.get_yaxis().set_tick_params(which='both', direction='in')
    a0.get_xaxis().set_tick_params(which='both', direction='in')
    a0.tick_params(right=True, which='both', direction='in')
    a_top.get_yaxis().set_tick_params(which='both', direction='in')
    a_top.get_xaxis().set_tick_params(which='both', direction='in')

    fig.text(0.5,0, xlabel, ha='center', fontsize='large')
    fig.text(-0.02,0.5, '"Dip" obs. - calc. [$P_\mathrm{{s}}$]', va='center', rotation=90, fontsize='large')

    fig.tight_layout(h_pad=0, w_pad=0)

    savefig(fig, savpath, dpi=350)


def plot_brethren(outdir):

    from transitleastsquares import transitleastsquares

    # IDs to plot
    epic_ids = [
        '204143627',
        '204270520',
        '204321142',
        # '246969828', # maybe? also Taurus, double dips
        '246938594', # 'Taurus',
        '205483258' # RIK-210
    ]

    # see /Users/luke/Dropbox/proj/billy/doc/20200417_list_of_analogs.txt
    id_dict = {
        "204143627": 'USco',
        "204270520": 'USco', # 204270520 has a C15 LC, but it's the only one.
        "204321142": 'USco',
        "205046529": 'USco', # kind of messed up, b/c it has two components
        "205483258": 'USco',
        '204787516': 'USco',
        '246938594': 'Taurus',
        '246969828': 'Taurus',
        '247794636': 'Taurus',  # does some wild stuff
        '246676629': 'Taurus',
        '246682490': 'Taurus',
        '247343526': 'Taurus'
    }

    lc_dict = {}

    k2pkl = os.path.join(outdir, 'k2data.pkl')
    if not os.path.exists(k2pkl):
        for epic_id in epic_ids:

            lc_dict[epic_id] = {}

            if id_dict[epic_id] == 'USco':
                lcpath = glob(
                    '/Users/luke/Dropbox/proj/billy/data/analogs/hlsp_everest_k2_llc_{}-c02_kepler_v2.0_lc/*fits'.format(epic_id)
                )
            elif id_dict[epic_id] == 'Taurus':
                lcpath = glob(
                    '/Users/luke/Dropbox/proj/billy/data/analogs/hlsp_everest_k2_llc_{}-c13_kepler_v2.0_lc/*fits'.format(epic_id)
                )
            assert len(lcpath)==1

            hdul = fits.open(lcpath[0])
            time = hdul[1].data['TIME']
            flux = hdul[1].data['FLUX']
            qual = hdul[1].data['QUALITY']
            hdul.close()

            plt.close('all')
            fig = plt.figure(figsize=(16,4))
            plt.scatter(time, flux, c='k', s=3)
            outpath = os.path.join(outdir, epic_id+'_quicklook_lc.png')
            savefig(fig, outpath, dpi=200, writepdf=0)

            if id_dict[epic_id] == 'USco':
                sel = (time > 2065) # & (qual < 20000) # initial part of campaign 2 has some garbage points
                time, flux = time[sel], flux[sel]
            elif id_dict[epic_id] == 'Taurus':
                sel = (time > 2990) # & (qual < 20000) # initial part of campaign 2 has some garbage points
                time, flux = time[sel], flux[sel]
            else:
                raise NotImplementedError

            time, flux, _ = sigclip_magseries(time, flux, None, sigclip=[50,5],
                                              iterative=True, niterations=2,
                                              magsarefluxes=True)

            plt.close('all')
            fig = plt.figure(figsize=(16,4))
            plt.scatter(time, flux, c='k', s=3)
            outpath = os.path.join(outdir, epic_id+'_quicklook_lc_clean.png')
            savefig(fig, outpath, dpi=200, writepdf=0)

            from wotan import flatten

            # 48 cadences per day... 240 = 5 days. 300 = 6 days.
            window_length = 1201 if epic_id=="205483258" else 401
            flatten_lc, trend_lc = flatten(time, flux, method='medfilt',
                                           window_length=window_length,
                                           return_trend=True,)

            plt.close('all')
            fig = plt.figure(figsize=(16,4))
            plt.scatter(time, flatten_lc, c='k', s=3)
            outpath = os.path.join(outdir, epic_id+'_quicklook_lc_flat.png')
            savefig(fig, outpath, dpi=200, writepdf=0)

            if id_dict[epic_id] == 'USco':
                sel = (time > 2071) & (time < 2129)
                flux = flatten_lc[sel]
                time = time[sel]
            elif id_dict[epic_id] == 'Taurus':
                flux = flatten_lc
                pass
            else:
                raise NotImplementedError

            plt.close('all')
            fig = plt.figure(figsize=(16,4))
            plt.scatter(time, flux, c='k', s=3)
            plt.ylim((np.nanmean(flux)-4*np.nanstd(flux),
                      np.nanmean(flux)+4*np.nanstd(flux)))
            outpath = os.path.join(outdir, epic_id+'_quicklook_lc_final.png')
            savefig(fig, outpath, dpi=200, writepdf=0)

            plt.close('all')

            lc_dict[epic_id]['time'] = time
            lc_dict[epic_id]['flux'] = flux

            # now phase-fold it. save and bin.
            period_min, period_max = 0.35, 6
            pdm_d = periodbase.stellingwerf_pdm(time, flux, flux*1e-3,
                                                magsarefluxes=True,
                                                startp=period_min,
                                                endp=period_max,
                                                stepsize=1.0e-4, autofreq=True,
                                                normalize=False,
                                                phasebinsize=0.05,
                                                mindetperbin=9, nbestpeaks=5,
                                                periodepsilon=0.1,
                                                sigclip=None, nworkers=16,
                                                verbose=True)
            period = pdm_d['bestperiod']

            # NOTE: I tried TLS and BLS. They have this insane multithreading
            # bug on catalina:
            # https://github.com/matplotlib/matplotlib/issues/15410
            percentile_int = 50
            nearest_index = (
                abs(flux - np.percentile(flux, percentile_int,
                                         interpolation='nearest')).argmin()
            )
            t0 = time[nearest_index]

            HALFIDS = [
                '204143627',
                '204270520',
                '205483258'
            ]
            if np.any(np.in1d(np.array(HALFIDS),np.array(epic_id))):
                print('hi')
                t0 += period/2
            else:
                pass
                print('dude')
            lc_dict[epic_id]['t0'] = t0

            p_d = phase_magseries(
                time, flux, period, t0, wrap=True, sort=True
            )
            pb_d = phase_bin_magseries(
                p_d['phase'], p_d['mags'], binsize=0.01
            )

            lc_dict[epic_id]['period'] = period

            lc_dict[epic_id]['phase_d'] = p_d
            lc_dict[epic_id]['phasebin_d'] = pb_d

        with open(k2pkl, 'wb') as buff:
            pickle.dump(lc_dict, buff)

    lc_dict = pickle.load(open(k2pkl, 'rb'))

    pklpath = '/Users/luke/Dropbox/proj/billy/results/PTFO_8-8695_results/20200413_v0/PTFO_8-8695_transit_2sincosPorb_2sincosProt_phasefoldmap_points.pkl'
    ptfo_d = pickle.load(open(pklpath, 'rb'))

    ##########

    plt.close('all')

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8.5, 10), sharex=True)

    axs = axs.flatten()

    for ix, ax in enumerate(axs):

        if ix == 0:
            ident = 'PTFO 8-8695'
            phase = ptfo_d['orb_d']['phase']
            flux = ptfo_d['orb_d']['mags']
            period = ptfo_d['orb_d']['period']
            binphase = ptfo_d['orb_bd']['binnedphases']
            binflux = ptfo_d['orb_bd']['binnedmags']

        else:
            ident = epic_ids[ix-1]
            phase = lc_dict[ident]['phase_d']['phase']
            flux = lc_dict[ident]['phase_d']['mags'] - 1
            period = lc_dict[ident]['phase_d']['period']
            binphase =  lc_dict[ident]['phasebin_d']['binnedphases']
            binflux = lc_dict[ident]['phasebin_d']['binnedmags'] - 1

        ax.scatter(phase, flux, color='gray', s=2, alpha=0.8, zorder=4,
                   linewidths=0, rasterized=True)
        ax.scatter(binphase, binflux, color='black', s=8, alpha=1, zorder=5,
                   linewidths=0)

        txt = '$P$ = {:.5f}$\,$d'.format(period)
        props = dict(boxstyle='square', facecolor='white', alpha=0.9, pad=0.15,
                     linewidth=0)
        ax.text(0.98, 0.98, txt, ha='right', va='top', transform=ax.transAxes,
                bbox=props, zorder=6)

        txt = ident if 'PTFO' in ident else 'EPIC '+ident
        props = dict(boxstyle='square', facecolor='white', alpha=0.9, pad=0.15,
                     linewidth=0)
        ax.text(0.02, 0.98, txt, ha='left', va='top', transform=ax.transAxes,
                bbox=props, zorder=6)

        ax.set_xlim((-1, 1))
        ax.set_ylim((np.mean(flux)-3*np.std(flux), np.mean(flux)+3*np.std(flux)))

        format_ax(ax)

    fig.text(0.5,0, 'Phase', ha='center', fontsize='x-large')
    fig.text(0.,0.5, 'Relative flux', va='center', rotation=90,
             fontsize='x-large')

    fig.tight_layout(h_pad=0, w_pad=0)

    outpath = os.path.join(outdir, 'brethren.png')
    savefig(fig, outpath)
