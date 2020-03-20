import os, corner
import numpy as np, matplotlib.pyplot as plt
from datetime import datetime
from pymc3.backends.tracetab import trace_to_dataframe

from billy.convenience import flatten as bflatten

from astrobase.lcmath import phase_magseries, phase_bin_magseries

def plot_test_data(x_obs, y_obs, y_mod, modelid, outdir):
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
    plt.figure(figsize=(14, 4))
    plt.plot(x_obs, y_obs, ".k", ms=4, label="data")
    plt.plot(x_obs, y_MAP, lw=1)
    plt.ylabel("relative flux")
    plt.xlabel("time [days]")
    _ = plt.title("MAP model")
    fig = plt.gcf()
    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_sampleplot(m, outpath, N_samples=100):
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


def plot_splitsignal_map(m, outpath):
    """
    y_obs + y_MAP + y_rot + y_orb
    things at rotation frequency
    things at orbital frequency
    """
    fig, axs = plt.subplots(nrows=4, figsize=(14, 12), sharex=True)

    axs[0].set_ylabel('flux')
    axs[0].plot(m.x_obs, m.y_obs, ".k", ms=4, label="data")
    axs[0].plot(m.x_obs, m.map_estimate['mu_model'], lw=0.5, label='MAP',
                color='C0', alpha=1, zorder=5)

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
            axs[0].plot(m.x_obs, y_rot, lw=0.5, label='model '+f,
                        color='C{}'.format(ix+1), alpha=1, zorder=ix+3)
        if f == 'orb':
            y_orb = yval + y_tra
            axs[0].plot(m.x_obs, y_orb, lw=0.5, label='model '+f,
                        color='C{}'.format(ix+1), alpha=1, zorder=ix+3)

    axs[1].set_ylabel('flux-orb (rot)')
    axs[1].plot(m.x_obs, m.y_obs-y_orb, ".k", ms=4, label="data-orb")
    axs[1].plot(m.x_obs, m.map_estimate['mu_model']-y_orb, lw=0.5,
                label='model-orb', color='C0', alpha=1, zorder=5)

    axs[2].set_ylabel('flux-rot (orb)')
    axs[2].plot(m.x_obs, m.y_obs-y_rot, ".k", ms=4, label="data-rot")
    axs[2].plot(m.x_obs, m.map_estimate['mu_model']-y_rot, lw=0.5,
                label='model-rot', color='C0', alpha=1, zorder=5)

    axs[3].set_ylabel('flux-model')
    axs[3].plot(m.x_obs, m.y_obs-m.map_estimate['mu_model'], ".k", ms=4, label="data")
    axs[3].plot(m.x_obs, m.map_estimate['mu_model']-m.map_estimate['mu_model'],
                lw=0.5, label='model', color='C0', alpha=1, zorder=5)


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


def plot_phasefold_map(m, d, outpath):

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
    # corner plot of posterior samples
    trace_df = trace_to_dataframe(m.trace, varnames=list(true_d.keys()))
    truths = [true_d[k] for k in true_d.keys()]
    truths = list(bflatten(truths))
    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        truths=truths, title_fmt='.2g')
    savefig(fig, outpath, writepdf=0)


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
