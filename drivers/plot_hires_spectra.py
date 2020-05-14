from astropy.io import fits
from glob import glob
import numpy as np, matplotlib.pyplot as plt
from cdips_followup.spectools import plot_orders
import os

outdir = '../results/hires_spectra/plot_orders/'
specpaths = glob('../data/hires_spectra/deblazed/*117*fits')

for s in specpaths:
    idstring = os.path.basename(s).replace('.fits', '')
    plot_orders(s, outdir=outdir, idstring=idstring)
