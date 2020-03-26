import os
import billy.plotting as bp
from billy import __path__

REALID = 'PTFO_8-8695'
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
PLOTDIR = os.path.join(RESULTSDIR, '{}_results'.format(REALID))

bp.plot_periodogram(PLOTDIR, islinear=True)
