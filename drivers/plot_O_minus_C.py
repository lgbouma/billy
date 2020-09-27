'''
make O-C model figure
'''
import pandas as pd, numpy as np
import os
from numpy import array as nparr
from astrobase.timeutils import get_epochs_given_midtimes_and_period

import billy.plotting as bp

EPOCH = 2455543.94300
PERIOD = 0.448399

def get_data(
    datacsv='../data/WASP-18b_transits_and_TESS_times_O-C_vs_epoch_selected.csv',
    is_occultation=False,
    plongphasing=False
    ):

    df = pd.read_csv(datacsv, sep=';')

    # 'period', 'period_err', 't0_HJD', 'err_t0', 't0_BJD_TDB', 'reference',
    #    'where_I_got_time', 'comment'

    y = nparr(df.t0_BJD_TDB)
    sigma_y = nparr(df.err_t0)
    refs = nparr(df.reference)

    sel = np.isfinite(y) & np.isfinite(sigma_y)

    y, sigma_y, refs = y[sel], sigma_y[sel], refs[sel]

    if not plongphasing:
        x, _ = get_epochs_given_midtimes_and_period(
            y, PERIOD, t0_fixed=EPOCH, verbose=True
        )
    else:
        x, _ = get_epochs_given_midtimes_and_period(
            y, 0.4991110, t0_fixed=EPOCH, verbose=True
        )

    return x, y, sigma_y, refs


def main(xlim=None, ylim=None, include_all_points=False, ylim1=None,
         plongphasing=0, nodash=0, simplelabel=0):

    homedir = os.path.expanduser('~')
    basedir = os.path.join(homedir, 'Dropbox/proj/billy/')
    transitpath = (
        basedir+'data/ephemeris/PTFO_8-8695_manual_all.csv'
    )
    occpath = 'foo.csv'

    print('getting data from {:s}'.format(transitpath))
    x, y, sigma_y, refs = get_data(datacsv=transitpath,
                                   plongphasing=plongphasing)
    print('getting data from {:s}'.format(occpath))

    if os.path.exists(occpath):
        raise NotImplementedError

    linear_params = nparr([EPOCH,  PERIOD])
    if plongphasing:
        linear_params = nparr([EPOCH, 0.4991110])

    savpath = '../results/ephemeris/O_minus_C.png'
    ylabel = None
    if plongphasing:
        savpath = '../results/ephemeris/O_minus_C_plongphasing.png'
    if nodash:
        savpath = '../results/ephemeris/O_minus_C_nodash.png'
    if simplelabel:
        ylabel = 'Dip phase [$P_\mathrm{{s}}$]'
        savpath = '../results/ephemeris/O_minus_C_nodash_simplelabel.png'

    onlytransits = True
    bp.plot_O_minus_C(
        x, y, sigma_y, linear_params, refs, savpath=savpath,
        xlabel='Cycle number', ylabel=ylabel, xlim=xlim, ylim=ylim,
        include_all_points=include_all_points, ylim1=ylim1,
        onlytransits=onlytransits, plongphasing=plongphasing, nodash=nodash)

    if onlytransits:
        print('made the O-C plot with only transits')


if __name__=="__main__":

    # with all points
    ylim = None #[-5,5]
    xlim = [-2000, 7800]
    ylim1 = None #FIXME [-5,4.2]
    main(xlim=xlim, ylim=ylim, ylim1=ylim1)

    # with all points, but no dashed line
    ylim = None #[-5,5]
    xlim = [-2000, 7800]
    ylim1 = None #FIXME [-5,4.2]
    main(xlim=xlim, ylim=ylim, ylim1=ylim1, nodash=1, simplelabel=0)

    # with P_long phasing, to see
    ylim = None
    xlim = None
    ylim1 = None
    main(xlim=xlim, ylim=ylim, ylim1=ylim1, plongphasing=1)
