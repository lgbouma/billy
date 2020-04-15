import os
import billy.plotting as bp
from billy import __path__

RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
PLOTDIR = os.path.join(RESULTSDIR, 'cluster_membership')

bp.plot_hr(PLOTDIR)
