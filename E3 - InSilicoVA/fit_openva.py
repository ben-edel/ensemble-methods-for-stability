
import numpy as np
import pandas as pd
import pickle as pkl
import os, sys
import re

import warnings
warnings.filterwarnings('ignore')

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import r

# Import R packages
base = importr('base')
utils = importr('utils')
openva = importr('openVA')

# Load data
df = pd.read_csv('va_data_adult.csv', low_memory=False)

# Fix mixed types in columns
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str)
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except ValueError:
            pass

# CoD lists
cancers = ['Breast Cancer', 'Cervical Cancer', 'Colorectal Cancer', 'Esophageal Cancer', 
           'Leukemia/Lymphomas', 'Lung Cancer', 'Prostate Cancer', 'Stomach Cancer']
cardiovascular = ['Acute Myocardial Infarction', 'Stroke', 'Other Cardiovascular Diseases']
external = ['Bite of Venomous Animal', 'Drowning', 'Falls', 'Fires', 
            'Homicide', 'Suicide', 'Road Traffic', 'Other Injuries', 'Poisonings']
infectious = ['AIDS', 'Diarrhea/Dysentery', 'Malaria', 'Pneumonia', 'TB', 'Other Infectious Diseases']
other = ['Asthma', 'Cirrhosis', 'COPD', 'Diabetes', 'Epilepsy', 
         'Maternal', 'Renal Failure', 'Other Non-communicable Diseases']

def map_cause_string(code):
    if code in cancers:
        return 'Cancers'
    elif code in cardiovascular:
        return 'Cardiovascular'
    elif code in external:
        return 'External'
    elif code in infectious:
        return 'Infectious Diseases'
    else:
        return 'Other Non-Communicable'
    
cod_numeration = {
    'Cancers': 0,
    'Cardiovascular': 1,
    'External': 2,
    'Infectious Diseases': 3,
    'Other Non-Communicable': 4
}

# De-fragment
df = df.copy()

df['cod_category'] = df['gs_text34'].apply(map_cause_string)

# Train/test split by site
val_site = 'Dar'
training_set = df[df.site!=val_site].reset_index(drop=True)
validation_set = df[df.site==val_site].reset_index(drop=True)

run_type = sys.argv[1]
run_id = sys.argv[2]
run_path = os.path.join(run_type, run_id)
if not os.path.exists(run_path): 
    os.makedirs(run_path)

np.random.seed(int(run_id))

n = len(training_set)
m = int(n/2)

if run_type=='subbagging':
    inds = np.random.choice(n, m, replace=False)
    training_set = training_set.iloc[inds]
elif run_type=='loo':
    drop_ind = int(sys.argv[3])
    inds = np.delete(np.arange(n), drop_ind)
    training_set = training_set.iloc[inds]
else: 
    inds = np.arange(n)

with open(os.path.join(run_path, 'indices.pkl'), 'wb') as f:
    pkl.dump(inds, f)

with localconverter(robjects.default_converter + pandas2ri.converter):
    r_train = robjects.conversion.py2rpy(training_set)
    r_test = robjects.conversion.py2rpy(validation_set)

from rpy2.robjects import r

# Put r_train and r_test into the R global env
r.assign("train", r_train)
r.assign("test", r_test)

r('''
library(openVA)

fit_ins <- codeVA(data = test, data.type = "PHMRC", model = "InSilicoVA",
                  data.train = train, causes.train = "cod_category",
                  phmrc.type = "adult", convert.type = "fixed",
                  Nsim = 5000, auto.length = FALSE)

indiv_prob <- getIndivProb(fit_ins)

results_table <- data.frame(
  as.data.frame(indiv_prob),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
''')

results_table_df = pandas2ri.rpy2py(r['results_table'])
res_df = results_table_df.rename(columns=cod_numeration)
res_df = res_df[sorted(res_df.columns)]

N = len(validation_set)
K = df["cod_category"].nunique()
w = np.zeros((N, K))

available_cols = [col for col in res_df.columns if col in range(K)]

for col in available_cols:
    w[:, col] = res_df[col].values

# Ensure probabilities sum to 1 (should already be true)
w = w/w.sum(1)[:, np.newaxis]

with open(os.path.join(run_path, 'weights.pkl'), 'wb') as f:
    pkl.dump(w, f)