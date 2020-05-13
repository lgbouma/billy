import pickle, os
import numpy as np, pandas as pd
from glob import glob

run_id = '20200513_v0'

pklpaths = glob(
    '/Users/luke/Dropbox/proj/billy/results/PTFO_8-8695_results/{}/*bicdict*'
    .format(run_id)
)
ds = []
for p in pklpaths:
    d = pickle.load(open(p,'rb'))
    ds.append(d)

df = pd.DataFrame(ds)

df = df.sort_values(by='BIC')

df['D_BIC'] = df.BIC - df.BIC.min()

df = df.drop('modelid', axis=1)

df['chisq'] = np.round(df.chisq, 1)
df['redchisq'] = np.round(df.redchisq, 3)
df['BIC'] = np.round(df.BIC, 1)
df['D_BIC'] = np.round(df.D_BIC, 1)

outpath = '../results/PTFO_8-8695_results/{}/bic_table_data.tex'.format(run_id)

df.to_latex(outpath, index=False)
print('wrote {}'.format(outpath))
