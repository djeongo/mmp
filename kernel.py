# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mmp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./"))

# Any results you write to the current directory are saved as output.
nrows = 80000
df_train = mmp.load_data.load_data('./train.csv', nrows)

mmp.preprocessing.convert_ver_to_int(df_train)
print(df_train.columns)

df_categorical = mmp.preprocessing.get_categorical(df_train)
df_categorical = mmp.preprocessing.clean_categorical(df_categorical)
index = df_categorical.index
#print(df_categorical.isnull().sum().sort_values(ascending=False))

#numpy array
onehot_categorical = mmp.preprocessing.encode_categorical(df_categorical)

df_float = mmp.preprocessing.get_float(df_train)
df_float = df_float.loc[index, :]


#X, y = mmp.preprocessing.get_training_set([df_float], df_train)
#mmp.analysis.classify_with_logistic_regression(X, y)

print(len(index), len(df_float))

X, y = mmp.preprocessing.get_training_set([df_float], [onehot_categorical], df_train.loc[index, 'HasDetections'])
mmp.analysis.classify_with_nn(X, y)
