from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import numpy as np


goF = pd.read_csv('../goF.csv', header=None, index_col=0)
goP = pd.read_csv('../goP.csv', header=None, index_col=0)
names = np.load('arrays/names.npy')

goP = goP.loc[names]
goF = goF.loc[names]
go = pd.concat((goF,goP), axis=1)

print("goF",goF[(goF.T == 0).all()].values.shape)
print("goP",goP[(goP.T == 0).all()].values.shape)
print("go",go[(go.T == 0).all()].values.shape)

goF.to_pickle('arrays/goF_df.pckl')
goP.to_pickle('arrays/goP_df.pckl')
go.to_pickle('arrays/go_df.pckl')
