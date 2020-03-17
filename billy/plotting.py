import os, corner
import numpy as np, matplotlib.pyplot as plt
from datetime import datetime
from pymc3.backends.tracetab import trace_to_dataframe

from billy.convenience import flatten as bflatten

def plot_test_data(x_obs, y_obs, y_mod, modelid, outdir):
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111, xlabel='x_obs', ylabel='y_obs',
                         title='Generated data and underlying model')
    ax.plot(x_obs, y_obs, 'x', label='sampled data')
    ax.plot(x_obs, y_mod, label='true regression line', lw=2.)
    plt.legend(loc=0)
    outpath = os.path.join(outdir, 'test_{}_data.png'.format(modelid))
    format_ax(ax)
    savefig(fig, outpath, writepdf=0)


def plot_MAP_data(x_obs, y_obs, y_MAP, outpath):
    plt.figure(figsize=(14, 4))
    plt.plot(x_obs, y_obs, ".k", ms=4, label="data")
    plt.plot(x_obs, y_MAP, lw=1)
    plt.ylabel("relative flux")
    plt.xlabel("time [days]")
    _ = plt.title("MAP model")
    fig = plt.gcf()
    savefig(fig, outpath, writepdf=0)


def plot_sampleplot(m, outpath, N_samples=100):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(m.x_obs, m.y_obs, ".k", ms=4, label="data", zorder=N_samples+1)
    ax.plot(m.x_obs, m.map_estimate['mu_model'], lw=0.5, label='MAP',
            zorder=N_samples+2, color='C1', alpha=1)

    np.random.seed(42)
    y_mod_samples = (
        m.trace.mu_model[np.random.choice(m.trace.mu_model.shape[0], N_samples,
                                          replace=False), :]
    )
    # np.random.choice(
    #     m.trace.mu_model, size=(N_samples, len(m.x_obs)), replace=True
    # )

    for i in range(N_samples):
        if i % 10 == 0:
            print(i)
        ax.plot(m.x_obs, y_mod_samples[i,:], color='C0', alpha=0.3,
                rasterized=True, lw=0.5)

    ax.set_ylabel("relative flux")
    ax.set_xlabel("time [days]")
    ax.legend(loc='best')
    savefig(fig, outpath, writepdf=0)


def plot_traceplot(m):
    # trace plot from PyMC3
    outpath = '../results/driver_results/test_{}_traceplot.png'.format(m.modelid)
    if not os.path.exists(outpath):
        plt.figure(figsize=(7, 7))
        pm.traceplot(m.trace[100:])
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close('all')


def plot_cornerplot(f,m):
    # corner plot of posterior samples
    trace_df = trace_to_dataframe(m.trace, varnames=list(f.true_d.keys()))
    truths = [f.true_d[k] for k in f.true_d.keys()]
    truths = list(bflatten(truths))
    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        truths=truths, title_fmt='.2g')
    outpath = '../results/driver_results/test_{}_corner.png'.format(m.modelid)
    savefig(fig, outpath, writepdf=0)


def savefig(fig, figpath, writepdf=True):
    fig.savefig(figpath, dpi=450, bbox_inches='tight')
    print('{}: made {}'.format(datetime.utcnow().isoformat(), figpath))

    if writepdf:
        pdffigpath = figpath.replace('.png','.pdf')
        fig.savefig(pdffigpath, bbox_inches='tight', rasterized=True, dpi=450)
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
