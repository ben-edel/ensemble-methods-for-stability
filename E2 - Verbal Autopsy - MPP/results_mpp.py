import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import os, re
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(0)

## custom color palette
lblue = (40/255,103/255,178/255)
cred  = (177/255, 4/255, 14/255)
nyllw = (245/255,213/255,71/255)# Naples yellow

def proj(w, j, eps):
    d = len(w)
    wj = w[j]
    w1 = np.sort(np.delete(w,j))[::-1]
    means = (w[j] + np.hstack([0,np.cumsum(w1)]))/np.arange(1,d+1)
    shrink_val = means - (eps/np.sqrt(2))/np.arange(1,d+1)
    ind = np.min(np.where(np.hstack([w1, -float('Inf')]) < shrink_val))
    w_proj = w.copy()
    if ind>0:
        w_proj[w_proj > shrink_val[ind]] = shrink_val[ind]
        w_proj[j] = shrink_val[ind] + eps/np.sqrt(2)
    return(w_proj, np.linalg.norm(w-w_proj))

def kstar(w, eps):
    d = len(w)
    k = 0
    o = np.argsort(-w)
    for i in range(d):
        j = o[i]
        _, dist = proj(w, j, eps)
        if dist < eps:
            k+=1
        else:
            break
    return k

def s_argmax(w):
    return(set([np.argmax(w)]))

def s(w, eps):
    k = kstar(w, eps)
    order = np.argsort(w)[::-1]
    return(set(order[:k]))

def s_topk(w, k=2):
    return(set(np.argsort(w)[::-1][:k]))

def s_thresh(w, t=.8):
    o = np.argsort(w)[::-1]
    k = np.where(np.cumsum(w[o]) >= t)[0].min()+1
    return(set(o[:k]))

def s_ndd(w, beta=1):
    L = len(w)
    o = np.argsort(w)[::-1]
    w_= w[o]
    i = 0 
    Delta = 1 - (beta**2+1)/(beta**2+np.arange(1, L+1))*np.cumsum(w_)
    I = np.where(Delta[:-1] - Delta[1:] < 0)[0]
    
    if I.size > 0:
        return(set(o[:(I.min()+1)]))
    else:
        return(set(o))
    
def g(s, delta=1.6, gamma=.6):
    return(delta/s - gamma/s**2)

def s_u(w, gvec):
    L = len(w)
    o = np.argsort(w)[::-1]
    w_= w[o]
    i = 0 
    P = np.cumsum(w_)
    U = gvec*P
    I = np.where(U[:-1] - U[1:] > 0)[0]
    if I.size > 0:
        return(set(o[:(I.min()+1)]))
    else:
        return(set(o))
    
def fun(w, method):
    if method=='inflate':
        return(s(w, EPS))
    elif method=='argmax':
        return(s_argmax(w))
    elif method=='topk':
        return(s_topk(w))
    elif method=='threshold':
        return(s_thresh(w))
    elif method=='ndd':
        return(s_ndd(w))
    elif method=='SVBOP-u65':
        seq = np.arange(1,len(w)+1)
        return(s_u(w, g(seq)))
    elif method=='SVBOP-u80':
        seq = np.arange(1,len(w)+1)
        return(s_u(w, g(seq, 2.2, 1.2)))
    elif method=='prec':
        seq = np.arange(1,len(w)+1)
        return(s_u(w, 1/seq))
    
def u50(y, S):
    k = len(S)
    return((y in S)*1./k)

def u65(y, S):
    k = len(S)
    return((y in S)*(1.6/k - .6/k**2))

def u80(y, S):
    k = len(S)
    return((y in S)*(2.2/k - 1.2/k**2))

### LOAD DATA ###

df = pd.read_csv('va_data_adult.csv', low_memory=False)

# Fix mixed types in columns
for col in df.columns:
    if df[col].dtype == object:
        unknown_values = ['nan', '', "Don't Know", "Unknown", "Refused to Answer"]
        df[col] = df[col] = df[col].astype(str).str.strip().replace(unknown_values, np.nan)
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

df = df.copy()

# Apply it to your DataFrame
df['cod_category'] = df['gs_text34'].apply(map_cause_string)
df['cause_of_death'] = df['cod_category'].map(cod_numeration)

# CoD cols to keep
guiding_cols = ['newid', 'site', 'cod_category', 'cause_of_death']

# Find the position of first medical survey column
starting_pos = 71

# Get all columns after excluding newid
feature_list = df.columns[starting_pos:].tolist()

# Remove guiding columns
feature_list = [x for x in feature_list if x not in guiding_cols]

# Remove columns that start with 'c' or 'a' and end in a letter as these are problematic
feature_list = [col for col in feature_list if not re.match(r'^[ac]\d.*_.*[a-zA-Z]$', col)]

# Combine the columns we want to keep
cols_to_keep = guiding_cols + feature_list

# Create base df_numerized with selected columns
df_temp = df[cols_to_keep].copy()

# Start with the columns we keep as-is
df_numerized = df_temp[guiding_cols].copy()

# Process each column that needs to be converted to binary
columns_to_process = [col for col in df_temp.columns if col not in guiding_cols]

dfs_to_concat = []
for col in columns_to_process:
    unique_values = df_temp[col].unique()
    values_count = len(unique_values)
    if df_temp[col].dtype == object:
        # If more than unique value limit, do not convert to binary for every value
        unique_limit = 3
        if values_count > unique_limit:
            temp_cols = {}
            top_values = list(
                df_temp[col]
                .dropna()
                .value_counts()
                .nlargest(unique_limit)
                .index
            )
            for val in top_values:
                colname = f"{col}_{str(val).strip().lower()}"
                temp_cols[colname] = (df_temp[col] == val).astype(int)
            #temp_cols[f"{col}_other"] = (~df_temp[col].isin(top_values)).astype(int)

            if temp_cols:
                dfs_to_concat.append(pd.DataFrame(temp_cols))
        else:
            binaries = pd.get_dummies(df_temp[col], prefix=col, drop_first=False, dummy_na=True).astype(int)
            # Add the binary columns to input DataFrame
            dfs_to_concat.append(binaries)
    else:
        temp_cols = {}
        temp_cols[col] = df_temp[col]
        if temp_cols:
            dfs_to_concat.append(pd.DataFrame(temp_cols))


df_numerized = pd.concat([df_numerized] + dfs_to_concat, axis=1).reset_index(drop=True)
# Remove any columns that are constant
df_numerized = df_numerized.loc[:, df_numerized.nunique(dropna=False) > 1]

target_col = 'cause_of_death'
numeric_feature_cols = [col for col in df_numerized.columns if col not in guiding_cols]
survey_binary_feature_cols = [col for col in df_numerized.columns if col not in guiding_cols and not col.startswith('word_')]

feature_cols = survey_binary_feature_cols

# Train / validation split
validation_site = 'Dar'
training_set = df_numerized[df_numerized.site!=validation_site].reset_index(drop=True)
validation_set = df_numerized[df_numerized.site==validation_site].reset_index(drop=True)

training_set = training_set[feature_cols + [target_col]]
validation_set = validation_set[feature_cols + [target_col]]

### ### ###

n = len(training_set)
K = training_set[target_col].nunique()
Y = validation_set[target_col].values

### ### ###

# load full
fp = 'full/1'
with open(os.path.join(fp, 'weights.pkl'), 'rb') as f:
    w_full = pkl.load(f)
n = len(training_set)
N, K = w_full.shape

# load perturbations
fp = 'perturbed'
RUN_IDs = [fn for fn in os.listdir(fp) if '_' not in fn and '.' not in fn]
RUN_IDs = [fn for fn in RUN_IDs 
           if os.path.exists(os.path.join(fp, fn, 'weights.pkl'))]
N_bs = len(RUN_IDs)

w_bs = np.zeros((N_bs, N, K))
for b, ID in enumerate(RUN_IDs):
    with open(os.path.join(fp, ID, 'weights.pkl'), 'rb') as f:
        w_bs[b] = pkl.load(f)

# Load subbagging full predictions
fp = 'subbagging/0'
SB_IDs = sorted([fn for fn in os.listdir(fp) 
                 if os.path.exists(os.path.join(fp, fn, 'weights.pkl'))],
                key=lambda x: int(x))
w_subbagging = np.zeros((len(SB_IDs), N, K))
for b, ID in enumerate(SB_IDs):
    with open(os.path.join(fp, ID, 'weights.pkl'), 'rb') as f:
        w_subbagging[b] = pkl.load(f)

w_subbag = w_subbagging.mean(axis=0)

# Load subbagging predictions for each perturbation
fp = 'subbagging'
w_subbag_bs = np.zeros((N_bs, N, K))
for b, run_id in enumerate(RUN_IDs):
    fp_sb_bs = os.path.join('subbagging', run_id)
    SB_IDs_bs = sorted([fn for fn in os.listdir(fp_sb_bs)
                        if os.path.exists(os.path.join(fp_sb_bs, fn, 'weights.pkl'))],
                        key=lambda x: int(x))
    arr = np.zeros((len(SB_IDs_bs), N, K))
    for i, ID in enumerate(SB_IDs_bs):
        with open(os.path.join(fp_sb_bs, ID, 'weights.pkl'), 'rb') as f:
            arr[i] = pkl.load(f)
    w_subbag_bs[b] = arr.mean(axis=0)

METHODS = ['argmax', 'inflate']
WT_TYPES = ['base', 'subbagging']
EPS = 0.05
def dd():
    return defaultdict(list)
results = defaultdict(dd)

for j in tqdm(range(N)): 
    for wt_type in WT_TYPES:
        if wt_type == 'base':
            w = w_full[j]
        else:
            w = w_subbag[j]

        for method in METHODS:
            S = fun(w, method)
            ct_dj = 0
            ct_oc = 0
            ct_jac = 0
            ct_euc = 0
            bs_prec = 0
            for i in range(N_bs):
                if wt_type == 'base':
                    w_ = w_bs[i, j]
                else:
                    w_ = w_subbag_bs[i, j]
                S_ = fun(w_, method)
                ct_euc += np.linalg.norm(w - w_)
                ct_dj += 0 if (S&S_) else 1
                ct_oc += 1 - (len(S & S_) / min(len(S), len(S_))) # Overlap coefficient
                ct_jac += 1 - (len(S & S_) / len(S | S_)) # Jaccard
                bs_prec += (set([Y[j]]) == S_)        
            results['instability_disjoint'][(method, wt_type)].append(ct_dj/N_bs)
            results['instability_overlap'][(method, wt_type)].append(ct_oc/N_bs)
            results['instability_jaccard'][(method, wt_type)].append(ct_jac/N_bs)
            results['instability_euclidean'][(method, wt_type)].append(ct_euc/N_bs)
            results['accuracy'][(method, wt_type)].append((set([Y[j]]) == S))
            results['accuracy_loo'][(method, wt_type)].append(bs_prec/N_bs)
            results['coverage'][(method, wt_type)].append((Y[j] in S))
            results['size'][(method, wt_type)].append(len(S))

formal_names = {
    'argmax' : 'argmax',
    'inflate' : 'argmax$^\\varepsilon$',
    'topk' : '$\\text{top-}2$'
}

algo_labels = {
    'base':'$\\mathcal{A}$',
    'subbagging':'$\\widetilde{\\mathcal{A}}_m$',
}

metrics = ['accuracy', 'coverage', 'instability_euclidean', 'instability_overlap', 'size']
summarize = lambda x : (np.mean(x), np.std(x)/np.sqrt(len(x)))

def tee(s, f):
    print(s, end='')
    f.write(s)

with open('results_table.txt', 'w') as f:
    tee("""
\\rowcolors{2}{gray!25}{white}
\\begin{table}[!htbp]
\\centering
\\resizebox{\\textwidth}{!}{%
    {\\renewcommand{\\arraystretch}{1.5} 
    \\begin{tabular}{c c c c c c c c}
    \\rowcolor{gray!60}
    Sel. rule & Algo & $\\beta_{\\text{accuracy}}\\nearrow$ & $\\beta_{\\text{coverage}}\\nearrow$ & $\\beta_{\\text{eucl-instability}}\\searrow$ & $\\beta_{\\text{mean-instability}}\\searrow$ & $\\beta_{\\text{max-instability}}\\searrow$ & $\\beta_{\\text{size}}\\searrow$ \\\\\\hline
""", f)

    for method in METHODS:
        for wt_type in ['base', 'subbagging']:
            tee('\t ' + formal_names[method] + ' & ' +
                ('$\\widetilde{\\mathcal{A}}_m$' if wt_type=='subbagging' else '$\\mathcal{A}$') + ' & ', f)
            for i, metric in enumerate(metrics):
                if metric == 'instability_overlap':
                    tee('%.3f (%.3f)' % summarize(results[metric][(method, wt_type)]), f)
                    tee(' & ', f)
                    tee('%.3f' % np.max(results[metric][(method, wt_type)]), f)
                else:
                    tee('%.3f (%.3f)' % summarize(results[metric][(method, wt_type)]), f)
                if i < len(metrics) - 1:
                    tee(' & ', f)
            tee(' \\\\\n', f)

    tee(f"""
\\end{{tabular}}}}
    }}
    % EPS = {EPS}, n = {n}, N = {N}, val_site = {validation_site}
    \\medskip
    \\caption{{Add caption here.}}
    \\label{{tab:results-extended}}
\\end{{table}}\n""", f)

from tueplots import bundles

# INSTABILITY PLOT

plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update({'font.size' : 10,
                    'axes.titlesize': 10,
                    'lines.linewidth' : 2,})

def plot_sf(arr, c, lab=None, ls='-'):
    size = len(arr)
    plt.step(np.sort(arr)[::-1], np.arange(size)/size, 
             where='post', c=c, label=lab, ls=ls)
def plot_sf(arr, c, lab=None, ls='-'):
    size = len(arr)
    plt.step(np.hstack([1, np.sort(arr)[::-1]]), 
             np.hstack([1/(2*size), np.arange(1,1+size)/size]), 
             where='post', c=c, label=lab, ls=ls)
    
plt.figure(figsize=(4.5,2.781152949374527))

cs = [cred, lblue, nyllw, 'g', 'k']
    
plot_sf(results['instability_overlap'][('argmax','base')], cs[0],  
        formal_names['argmax'] + '$\circ \mathcal{A}$', '--')
#plot_sf(results['instability_overlap'][('inflate','base')], cs[1],  
#        formal_names['inflate'] + '$\circ \mathcal{A}$', '--')
plot_sf(results['instability_overlap'][('argmax','subbagging')], cs[0], 
        formal_names['argmax'] + '$\circ \widetilde{\\mathcal{A}}_m$', '-')
#plot_sf(results['instability_overlap'][('inflate','subbagging')], cs[1], 
#        formal_names['inflate'] + '$\circ \widetilde{\\mathcal{A}}_m$', '-')

plt.semilogy()
# plt.loglog()
plt.legend(loc='upper right')

plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')

plt.xticks([0, .5, 1])

plt.xlabel('$\\delta$', fontsize=12)
plt.ylabel('$\\frac{1}{N}\sum_{j\in [N]}1\\{\\delta_j > \\delta\\}$', 
           fontsize=12, labelpad=5)

L = 10

plt.ylim([(2/3)/N,2])
plt.xlim([-.005,1.01])

plt.title('Stability Comparison')
plt.tight_layout()

plt.savefig('instability_plot.pdf')
plt.show()

### Done ###