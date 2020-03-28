"""
TESS image (x's numbered)
DSS2 Red image.
"""

import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from billy.plotting import plot_scene

def main():
    # J2015.5 gaia
    ra = 81.28149227428
    dec = 1.57343091124

    c_obj = SkyCoord(ra, dec, unit=(u.deg), frame='icrs')

    hdul = fits.open(
        '../data/PTFO_8-8695/'
        'tess2018349182459-s0006-0000000264461976-0126-s/'
        'tess2018349182459-s0006-0000000264461976-0126-s_tp.fits'
    )

    img_wcs = wcs.WCS(hdul[2].header)

    d = hdul[1].data
    mean_img = np.nansum(d["FLUX"], axis=0) / d["FLUX"].shape[0]

    bkgd_mask = (hdul[2].data == 37)
    ap_mask = (hdul[2].data == 43)

    outpath = '../results/PTFO_8-8695_results/scene.png'
    plot_scene(c_obj, img_wcs, mean_img, outpath, ap_mask=ap_mask,
               bkgd_mask=bkgd_mask)


if __name__ == "__main__":
    main()
