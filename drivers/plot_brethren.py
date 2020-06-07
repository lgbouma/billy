import os
import billy.plotting as bp
from billy import __path__

RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')

# # exploration options
# PLOTDIR = os.path.join(RESULTSDIR, 'brethren_taurus')
# PLOTDIR = os.path.join(RESULTSDIR, 'brethren_usco')

# final plot
PLOTDIR = os.path.join(RESULTSDIR, 'brethren')

bp.plot_brethren(PLOTDIR)
bp.plot_brethren(PLOTDIR, twobythree=1)
