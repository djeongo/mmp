import pandas as pd
import mmp

def init(train_csv):
  # define dtype of each column
  df = pd.read_csv(train_csv, nrows=10)
  mmp.columns = list(df.columns)
  mmp.columns.remove('MachineIdentifier')
  mmp.columns.remove('OsBuildLab')
  mmp.columns.remove('PuaMode')
  mmp.columns.remove('Census_ProcessorClass')
  mmp.columns.remove('DefaultBrowsersIdentifier')
  mmp.columns.remove('Census_IsFlightingInternal')
  mmp.columns.remove('Census_InternalBatteryType')
  mmp.columns.remove('Census_ThresholdOptIn')
  mmp.columns.remove('Census_IsWIMBootEnabled')
  mmp.columns.remove('SmartScreen') # Too many NaN, 3177011
  mmp.columns.remove('OrganizationIdentifier') # Too many NaN, 2751518
  
  # log vals 
  NUM_VER = 4 # number of numeric values in version strings
  log_vals ={
      'AvSigVersion': {0: 0.0, 1: 3.0, 2: 4.0, 3: 0},
      'AppVersion': {0: 1.0, 1: 2.0, 2: 5.0, 3: 5.0},
      'EngineVersion': {0: 0.0, 1: 0.0, 2: 5.0, 3: 1.0},
      'Census_OSVersion': {0: 1.0, 1: 0, 2: 5.0, 3: 5.0},
      'OsVer': {0: 1.0, 1: 1.0, 2: 2.0, 3: 2.0}}
  
  def ver_to_int(row, col):
      try:
          tokens = [int(x) for x in row.split('.')]
      except:
          return 0
      int_ver = 0
      cum_mult = 0
      for i in range(NUM_VER-1,-1,-1):
          int_ver += tokens[i]*(10**cum_mult)
          cum_mult += log_vals[col][i]
      return int(int_ver)
  
  def ver_to_int_AvSigVersion(row): return ver_to_int(row, 'AvSigVersion')
  def ver_to_int_AppVersion(row): return ver_to_int(row, 'AppVersion')
  def ver_to_int_EngineVersion(row): return ver_to_int(row, 'EngineVersion')
  def ver_to_int_Census_OSVersion(row): return ver_to_int(row, 'Census_OSVersion')
  def ver_to_int_OsVer(row): return ver_to_int(row, 'OsVer')
  
  converters = {
      'AvSigVersion': ver_to_int_AvSigVersion,
      'AppVersion': ver_to_int_AppVersion,
      'EngineVersion': ver_to_int_EngineVersion,
      'Census_OSVersion': ver_to_int_Census_OSVersion,
      'OsVer': ver_to_int_OsVer}


def load_data(train_csv, nrows):
  init(train_csv)
  df_train = pd.read_csv(train_csv,
                       dtype=mmp.dtypes,
                       usecols=mmp.columns,
#                        converters=converters,
                       nrows=nrows
                      )
  return df_train
