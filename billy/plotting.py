import os
import numpy as np, matplotlib.pyplot as plt
from datetime import datetime

def plot_test_data(x_obs, y_obs, y_mod, modelid, outdir):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, xlabel='x_obs', ylabel='y_obs',
                         title='Generated data and underlying model')
    ax.plot(x_obs, y_obs, 'x', label='sampled data')
    ax.plot(x_obs, y_mod, label='true regression line', lw=2.)
    plt.legend(loc=0)
    outpath = os.path.join(outdir, 'test_{}_data.png'.format(modelid))
    format_ax(ax)
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
