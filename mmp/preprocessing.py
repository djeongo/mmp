import pandas as pd
import numpy as np
from sklearn import preprocessing
import mmp

# Convert version strings to integers
NUM_VER = 4 # number of numeric values in version strings
version_cols = ['AvSigVersion',
                'AppVersion',
                'EngineVersion',
                'Census_OSVersion',
                'OsVer'
               ]
max_vals = {}
log_vals = {}

def safe_split(row):
    try:
        tokens = [int(x) for x in row.split('.')]
        return tokens
    except:
        return [0]*NUM_VER


def compute_max_vals(df_train):
    df_split = pd.DataFrame() # Dataframe to contain version strings split into individual numbers
    for col in version_cols:
        df_split[col] = df_train[col].astype('object').apply(safe_split)
        for i in range(NUM_VER):
            df_split[col+str(i)] = df_split[col].apply(lambda x: x[i])
            if col not in max_vals:
                max_vals[col] = {}
            max_vals[col][i] = max(df_split[col+str(i)])

    # find max value and log value for each number in string version
    for col in version_cols:
        print(col)
        log_vals[col] = {}
        for i in range(NUM_VER):
            val = max_vals[col][i]
            log_val = np.ceil(np.log10(val))
            log_val = 0 if np.isinf(log_val) else log_val # set to 0 if infinity
            print("val={:8}, ceil(log10(val))={:8}".format(val, log_val))
            log_vals[col][i] = log_val



version_cols_int = []

def convert_ver_to_int(df_train):
  def ver_to_int(row, col):
      tokens = safe_split(row)
      int_ver = 0
      cum_mult = 0
      for i in range(NUM_VER-1,-1,-1):
          int_ver += tokens[i]*(10**cum_mult)
          cum_mult += log_vals[col][i]
      return int(int_ver)

  compute_max_vals(df_train)

  for col in version_cols:
    df_train[col+'_int'] = df_train[col].apply(ver_to_int, col=col).astype(np.float32)
    version_cols_int.append(col+'_int')

def get_categorical(df_train):
  """Get categorical columns

  Returns:
    A Pandas Dataframe containing only categorical columns

  """
  df_float = df_float = df_train.select_dtypes(include=['float16', 'float32'])
  df_float_to_cat = df_float[mmp.dtypes_float_to_cat].astype('category')
  df_categorical = df_train.select_dtypes(include=['category']).drop(version_cols, axis=1)
  df_categorical = pd.concat([df_categorical, df_float_to_cat], axis=1)
  
  print("categorical cols: ", len(df_categorical.columns))

  return df_categorical

def clean_categorical(df_categorical):
  """Remove categorical columns with NaN
  """
  return df_categorical.dropna(axis=0)
#  dict_categorical_clean = {}
#
#  for col in df_categorical:
#    series = df_categorical[col]
#    series = series.cat.add_categories(["missing"])
#    series.fillna("missing", inplace=True)
#    dict_categorical_clean[col] = series
#    
#  return pd.DataFrame(dict_categorical_clean)

def encode_categorical(df_categorical):
  """Encode categoircal columns

  Returns:
    A Pandas Dataframe containing one-hot encoded categorical columns
  """
  enc_ordinal = preprocessing.OneHotEncoder()

  onehot_encoded = enc_ordinal.fit_transform(df_categorical)

  print("one-hot encoded cols: ", onehot_encoded.shape[1])

  return onehot_encoded

def get_float(df_train):
  """Get float columns

  Returns:
    A Pandas Dataframe containing only float16 and float32 columns
  """
  df_float = df_train.select_dtypes(include=['float16', 'float32'])
  # drop columns that should really be categorical
  df_float = df_float.drop(mmp.dtypes_float_to_cat, axis=1)

  print("float cols: ", len(df_float.columns))

  return df_float

def get_training_set(dfs, arrays, y):
  """
  Args:
    df: A list of Pandas Dataframes each containing a subset of columns to use
    arrays: A list of Numpy arrays
    df_train: A Pandas Dataframe with 'HasDetections' column

  Returns:
    X: A Pandas Dataframe of input features
    y: A Pandas Series of output
  """
  # concatenate data frames
  df = pd.concat(dfs, axis=1)
  df_clean = df.dropna()
  index = df_clean.index
  y = y.loc[index]
  print("Dropped {:0.2f}% of rows".format((1-len(df_clean)/len(df))*100))
  X = df_clean
  
  # concatenate numpy arrays

  return X, y
