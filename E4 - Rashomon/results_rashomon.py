import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import os, re
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

import argparse

parser = argparse.ArgumentParser(description="Summarize Rashomon experiment results.")
parser.add_argument('dataset', choices=['synthetic', 'heloc'],
                     help="Which dataset's results to load and summarize.")
args = parser.parse_args()
dataset = args.dataset

if dataset == "synthetic":
    np.random.seed(0)

    X, y = make_classification(
        n_samples=300,
        n_features=15,        # expands Rashomon set
        n_informative=4,      # actually matter
        n_redundant=6,        # correlated with informative
        n_repeated=1,
        n_clusters_per_class=1,
        class_sep=0.25,        # low separability
        weights=[0.85, 0.15], # imbalance
        flip_y=0.1,          # label noise
        random_state=0
    )

    X = StandardScaler().fit_transform(X)

    # Split before any boundary-based manipulation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Thin uncertain training points to leave boundary region underspecified
    lr = LogisticRegression(C=1e6).fit(X_train_raw, y_train)
    probs = lr.predict_proba(X_train_raw)[:, 1]
    keep = np.abs(probs - 0.5) > 0.2
    X_train_raw = X_train_raw[keep]
    y_train = y_train[keep]

    # Inject ambiguous test points in continuous space before converting to DataFrame
    n_ambig = 500
    ambig_X = np.random.uniform(-1, 1, size=(n_ambig, X.shape[1]))
    ambig_y = (lr.decision_function(ambig_X) > 0).astype(int)
    X_test_raw = np.vstack([X_test_raw, ambig_X])
    y_test = np.concatenate([y_test, ambig_y])

    # Convert to DataFrames
    X_train = pd.DataFrame(X_train_raw, columns=[f"x{i}" for i in range(X_train_raw.shape[1])])
    X_test = pd.DataFrame(X_test_raw, columns=[f"x{i}" for i in range(X_test_raw.shape[1])])

    target_col = 'target'

    training_set = X_train.reset_index(drop=True)
    training_set[target_col] = y_train.reset_index(drop=True) if hasattr(y_train, 'reset_index') else y_train

    validation_set = X_test.reset_index(drop=True)
    validation_set[target_col] = y_test

elif dataset == "heloc":
    df = pd.read_csv('heloc.csv')

    X = df.drop(columns=['estimate_of_risk'])
    y = (df['estimate_of_risk'] >= 65).astype(int).values  # 1=Good, 0=Bad

    target_col = 'target'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    training_set = X_train.reset_index(drop=True)
    training_set[target_col] = y_train
    validation_set = X_test.reset_index(drop=True)
    validation_set[target_col] = y_test
else:
    raise ValueError(f"Unknown dataset: {dataset}")

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

# load perturbed
fp = 'perturbed'
RUN_IDs = [fn for fn in os.listdir(fp) if '_' not in fn and '.' not in fn]
RUN_IDs = [fn for fn in RUN_IDs 
           if os.path.exists(os.path.join(fp, fn, 'weights.pkl'))]
N_bs = len(RUN_IDs)

# get the perturbed ids 
ns = []
for ID in RUN_IDs:
    with open(os.path.join(fp, ID, 'indices.pkl'), 'rb') as f:
        ns.append(list(set(range(n)).difference(set(pkl.load(f))))[0])

w_bs = np.zeros((N_bs, N, K))
for b, ID in enumerate(RUN_IDs):
    with open(os.path.join(fp, ID, 'weights.pkl'), 'rb') as f:
        w_bs[b] = pkl.load(f)

# Load Rashomon full predictions
fp = 'rashomon/0'
with open(os.path.join(fp, 'weights.pkl'), 'rb') as f:
    w_rashomon = pkl.load(f)

# Load Rashomon perturbed predictions
fp = 'rashomon'
w_rashomon_bs = np.zeros((N_bs, N, K))
for b, ID in enumerate(RUN_IDs):
    with open(os.path.join(fp, ID, 'weights.pkl'), 'rb') as f:
        w_rashomon_bs[b] = pkl.load(f)

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
WT_TYPES = ['base', 'subbagging', 'rashomon']
EPS = 0.1
def dd():
    return defaultdict(list)
results = defaultdict(dd)

for j in tqdm(range(N)): 
    for wt_type in WT_TYPES:
        if wt_type == 'base':
            w = w_full[j]
        elif wt_type == 'rashomon':
            w = w_rashomon[j]
        elif wt_type == 'subbagging':
            w = w_subbag[j]

        for method in METHODS:
            S = fun(w, method)
            ct_dj = 0
            ct_oc = 0
            ct_jac = 0
            ct_euc = 0
            loo_prec = 0
            for i in range(N_bs):
                if wt_type == 'base':
                    w_ = w_bs[i, j]
                elif wt_type == 'rashomon':
                    w_ = w_rashomon_bs[i, j]
                elif wt_type == 'subbagging':
                    w_ = w_subbag_bs[i, j]
                S_ = fun(w_, method)
                ct_euc += np.linalg.norm(w - w_)
                ct_dj += 0 if (S&S_) else 1
                ct_oc += 1 - (len(S & S_) / min(len(S), len(S_))) # Overlap coefficient
                ct_jac += 1 - (len(S & S_) / len(S | S_)) # Jaccard
                loo_prec += (set([Y[j]]) == S_)                          
            results['instability_disjoint'][(method, wt_type)].append(ct_dj/N_bs)
            results['instability_overlap'][(method, wt_type)].append(ct_oc/N_bs)
            results['instability_jaccard'][(method, wt_type)].append(ct_jac/N_bs)
            results['instability_euclidean'][(method, wt_type)].append(ct_euc/N_bs)
            results['accuracy'][(method, wt_type)].append((set([Y[j]]) == S))
            results['accuracy_loo'][(method, wt_type)].append(loo_prec/N_bs)
            results['coverage'][(method, wt_type)].append((Y[j] in S))
            results['size'][(method, wt_type)].append(len(S))

formal_names = {
    'argmax' : 'argmax',
    'inflate' : 'argmax$^\\varepsilon$',
    'topk' : '$\\text{top-}2$',
    'threshold' : 'Thresholding $\\Gamma^*_{0.8}$',
    'ndd' : 'NDC for $F_1$',
    'SVBOP-u65' :  'SVBOP$_{u_{65}}$', 
    'SVBOP-u80' : 'SVBOP$_{u_{80}}$', 
    'prec' :  'SVBOP$_{u_{50}}$',
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

    algo_tex = {
        'base':       '$\\mathcal{A}$',
        'subbagging': '$\\widetilde{\\mathcal{A}}_m$',
        'rashomon':   '$\\widetilde{\\mathcal{A}}_{\\mathcal{R}}$',
    }

    for method in METHODS:
        for wt_type in ['base', 'subbagging', 'rashomon']:
            tee('\t ' + formal_names[method] + ' & ' + algo_tex[wt_type] + ' & ', f)
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
    % EPS = {EPS}
    \\medskip
    \\caption{{Add caption here.}}
    \\label{{tab:results-extended}}
\\end{{table}}\n""", f)

from tueplots import bundles

# INSTABILITY PLOT

plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update({'font.size': 10,
                     'axes.titlesize': 10,
                     'lines.linewidth': 2,
                     'text.usetex': False})

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

c_base      = cred
c_subbagging = lblue
c_rashomon  = nyllw

plot_sf(results['instability_overlap'][('argmax', 'base')],       c_base,       formal_names['argmax'] + '$\\circ \\mathcal{A}$',                          ls=(0, (3, 1)))   # dashed
plot_sf(results['instability_overlap'][('argmax', 'subbagging')], c_subbagging, formal_names['argmax'] + '$\\circ \\widetilde{\\mathcal{A}}_{m}$',          ls='-')            # solid
plot_sf(results['instability_overlap'][('argmax', 'rashomon')],   c_rashomon,   formal_names['argmax'] + '$\\circ \\widetilde{\\mathcal{A}}_{\\mathcal{R}}$', ls=(0, (3, 1, 1, 1)))   # dotted

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

# INFLUENCE PLOT

plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

plt.rcParams.update({
    'axes.labelsize': 13,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

influences_base = np.zeros(N_bs)
influences_subb = np.zeros(N_bs)
influences_rash = np.zeros(N_bs)

pred_full_base = np.argmax(w_full, axis=1)
pred_full_subb = np.argmax(w_subbag, axis=1)
pred_full_rash = np.argmax(w_rashomon, axis=1)

for idx in range(N_bs):
    pred_loo_base = np.argmax(w_bs[idx], axis=1)
    pred_loo_subb = np.argmax(w_subbag_bs[idx], axis=1)
    pred_loo_rash = np.argmax(w_rashomon_bs[idx], axis=1)
    influences_base[idx] = np.mean(pred_full_base != pred_loo_base)
    influences_subb[idx] = np.mean(pred_full_subb != pred_loo_subb)
    influences_rash[idx] = np.mean(pred_full_rash != pred_loo_rash)

fig, ax = plt.subplots()

for arr, c, lab, ls in [
    (influences_base, cred,  r'$\text{argmax}\circ\mathcal{A}$', (0, (3, 1))),
    (influences_subb, lblue, r'$\text{argmax}\circ\widetilde{\mathcal{A}}_m$', '-'),
    (influences_rash, nyllw, r'$\text{argmax}\circ\widetilde{\mathcal{A}}_{\mathcal{R}}$', (0, (3, 1, 1, 1))),
]:
    sorted_vals = np.sort(arr)
    n_pts = len(sorted_vals)
    x = np.concatenate([[0], sorted_vals])
    y = np.concatenate([[0], np.arange(1, n_pts+1) / n_pts])
    ax.step(x, y, where='post', color=c, label=lab, linestyle=ls, linewidth=1.5)

ax.set_xlabel(r'$\psi := \mathbb{P}_X\left(\hat{f}(X) \neq \hat{f}^{-i}(X)\right)$')
ax.set_ylabel(r'$\frac{1}{N}\sum_{i \in [N]} \mathbf{1}\{\psi_i \leq \psi\}$')
ax.legend(loc='lower right')
ax.spines[['bottom', 'left']].set_visible(True)
ax.spines[['top', 'right']].set_visible(True)
ax.set_ylim([0, 1])
ax.set_xlim(left=0)
ax.legend(loc='lower right', fontsize=10)

plt.savefig('influence_plot.pdf')
plt.show()

### Done ###