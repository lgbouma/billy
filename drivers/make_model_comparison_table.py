import pickle, os
import numpy as np, pandas as pd
from glob import glob

pklpaths = glob(
    '/Users/luke/Dropbox/proj/billy/results/PTFO_8-8695_results/20200413_v0/*bicdict*'
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

outpath = '../results/PTFO_8-8695_results/20200413_v0/bic_table_data.tex'
df.to_latex(outpath, index=False)
print('wrote {}'.format(outpath))
