import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import os, re
from collections import defaultdict

from scipy.special import softmax

import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(0)

## custom color palette
lblue = (40/255,103/255,178/255)
cred  = (177/255, 4/255, 14/255)
nyllw = (245/255,213/255,71/255)

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

target_col = 'cause_of_death'
df['cod_category'] = df['gs_text34'].apply(map_cause_string)
df[target_col] = df['cod_category'].map(cod_numeration)

# Train/test split by site
validation_site = 'Dar'
training_set = df[df.site!=validation_site].reset_index(drop=True)
validation_set = df[df.site==validation_site].reset_index(drop=True)

# Class labels
K = training_set[target_col].nunique()

### ### ###

Y = validation_set[target_col].values

# load full
fp = 'full/1'
with open(os.path.join(fp, 'weights.pkl'), 'rb') as f:
    w_full = pkl.load(f)
n = len(training_set)
m = int(n/2)
N, K = w_full.shape

fp = 'loo'
RUN_IDs = [fn for fn in os.listdir(fp) if '_' not in fn and '.' not in fn]
RUN_IDs = [fn for fn in RUN_IDs 
           if os.path.exists(os.path.join(fp, fn, 'weights.pkl'))]
N_loo = len(RUN_IDs)

# get the loo ids 
ns = []
for ID in RUN_IDs:
    with open(os.path.join(fp, ID, 'indices.pkl'), 'rb') as f:
        ns.append(list(set(range(n)).difference(set(pkl.load(f))))[0])

w_loo = np.zeros((N_loo, N, K))
for b, ID in enumerate(RUN_IDs):
    with open(os.path.join(fp, ID, 'weights.pkl'), 'rb') as f:
        w_loo[b] = pkl.load(f)

fp = 'subbagging'

RUN_IDs = []
for fn in tqdm(os.listdir(fp)):
    if (os.path.exists(os.path.join(fp, fn, 'weights.pkl')) 
        and os.path.exists(os.path.join(fp, fn, 'indices.pkl'))):
        RUN_IDs.append(fn)
# RUN_IDs = os.listdir(fp)

B_tot = len(RUN_IDs)#+1
B = min(B_tot, 250)

bags = np.zeros((B_tot, m), dtype=int)
w_bags = np.zeros((B_tot, N, K))
for b, ID in tqdm(enumerate(RUN_IDs)):
    with open(os.path.join(fp, ID, 'indices.pkl'), 'rb') as f:
        bags[b] = pkl.load(f)
    with open(os.path.join(fp, ID, 'weights.pkl'), 'rb') as f:
        w_bags[b] = pkl.load(f)
        
inds = np.random.choice(B_tot, B, replace=False)
w_subb = np.zeros(w_bags.shape[1:])
for b in tqdm(inds):
    w_subb += w_bags[b]
w_subb /= B

w_subb_= []
for i in ns:
    inds = np.where(np.all(bags!=i,1))[0]
    # Adjust to handle smaller number of bags
    B_sub = min(B, len(inds))
    print(f"Using {B_sub} bags for LOO point {i}.")
    w_subb_.append(np.mean(w_bags[np.random.choice(inds, B_sub, replace=False)], 0))
w_subb_ = np.array(w_subb_)

# Check spread of subbagg predictions
spread_per_j = np.zeros(N)
for j in range(N):
    dists = np.zeros(B_tot)
    for b in range(B_tot):
        dists[b] = np.linalg.norm(w_full[j] - w_bags[b, j])
    spread_per_j[j] = np.mean(dists)

METHODS = ['argmax', 'inflate']
EPS = 0.1
def dd():
    return defaultdict(list)
results = defaultdict(dd)

for j in tqdm(range(N)): 
    for wt_type in ['base', 'subbagged']:
        w = w_full[j] if wt_type=='base' else w_subb[j]

        for method in METHODS:
            S = fun(w, method)
            ct_dj = 0
            ct_oc = 0
            ct_jac = 0
            ct_euc = 0
            loo_prec = 0
            for i in range(N_loo):
                w_ = w_loo[i,j] if wt_type=='base' else w_subb_[i,j]
                S_ = fun(w_, method)
                ct_euc += np.linalg.norm(w - w_)
                ct_dj += 0 if (S&S_) else 1
                ct_oc += 1 - (len(S & S_) / min(len(S), len(S_))) # Overlap coefficient
                ct_jac += 1 - (len(S & S_) / len(S | S_)) # Jaccard
                loo_prec += (set([Y[j]]) == S_)                        
            results['instability_disjoint'][(method, wt_type)].append(ct_dj/N_loo)
            results['instability_overlap'][(method, wt_type)].append(ct_oc/N_loo)
            results['instability_jaccard'][(method, wt_type)].append(ct_jac/N_loo)
            results['instability_euclidean'][(method, wt_type)].append(ct_euc/N_loo)
            results['precision'][(method, wt_type)].append((set([Y[j]]) == S))
            results['precision_loo'][(method, wt_type)].append(loo_prec/N_loo)
            results['coverage'][(method, wt_type)].append((Y[j] in S))
            results['size'][(method, wt_type)].append(len(S))

#np.save(f'base_instability.npy', np.array(results['instability_overlap'][('argmax', 'base')]))

#print(f"Mean Euclidean Distance of Subbags: {np.mean(spread_per_j):.4f}")
#print(f"Max Euclidean Distance of Subbags: {np.max(spread_per_j):.4f}")

formal_names = {
    'argmax' : 'argmax',
    'inflate' : 'argmax$^\\varepsilon$',
    'topk' : '$\\text{top-}2$'
}

metrics = ['precision', 'coverage', 'instability_euclidean', 'instability_overlap', 'size']
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
    Sel. rule & Algo & $\\beta_{\\text{precision}}\\nearrow$ & $\\beta_{\\text{coverage}}\\nearrow$ & $\\beta_{\\text{eucl-instability}}\\searrow$ & $\\beta_{\\text{avg-instability}}\\searrow$ & $\\beta_{\\text{max-instability}}\\searrow$ & $\\beta_{\\text{size}}\\searrow$ \\\\\\hline
""", f)

    for method in METHODS:
        for wt_type in ['base', 'subbagged']:
            tee('\t ' + formal_names[method] + ' & ' +
                ('$\\widetilde{\\mathcal{A}}_m$' if wt_type=='subbagged' else '$\\mathcal{A}$') + ' & ', f)
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
    % EPS = {EPS}, B = {B}, n = {n}, val_site = {validation_site}
    % Mean Euclidean Distance of Subbags: {np.mean(spread_per_j):.4f}
    % Max Euclidean Distance of Subbags: {np.max(spread_per_j):.4f}
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
plot_sf(results['instability_overlap'][('inflate','base')], cs[1],  
        formal_names['inflate'] + '$\circ \mathcal{A}$', '--')
plot_sf(results['instability_overlap'][('argmax','subbagged')], cs[0], 
        formal_names['argmax'] + '$\circ \widetilde{\\mathcal{A}}_m$', '-')
plot_sf(results['instability_overlap'][('inflate','subbagged')], cs[1], 
        formal_names['inflate'] + '$\circ \widetilde{\\mathcal{A}}_m$', '-')

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

influences_base = np.zeros(N_loo)
influences_subb = np.zeros(N_loo)

pred_full_base = np.argmax(w_full, axis=1)
pred_full_subb = np.argmax(w_subb, axis=1)

for idx in range(N_loo):
    pred_loo_base = np.argmax(w_loo[idx], axis=1)
    pred_loo_subb = np.argmax(w_subb_[idx], axis=1)
    influences_base[idx] = np.mean(pred_full_base != pred_loo_base)
    influences_subb[idx] = np.mean(pred_full_subb != pred_loo_subb)

fig, ax = plt.subplots()

for arr, c, lab, ls in [
    (influences_base, cred,  r'$\text{argmax}\circ\mathcal{A}$', '--'),
    (influences_subb, lblue, r'$\text{argmax}\circ\widetilde{\mathcal{A}}_m$', '-'),
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

plt.savefig('influence_plot.pdf')
plt.show()

influences_base = np.zeros(N_loo)
influences_subb = np.zeros(N_loo)

for idx in range(N_loo):
    influences_base[idx] = np.mean(
        np.linalg.norm(w_full - w_loo[idx], axis=1)
    )
    influences_subb[idx] = np.mean(
        np.linalg.norm(w_subb - w_subb_[idx], axis=1)
    )

fig, ax = plt.subplots()

for arr, c, lab, ls in [
    (influences_base, cred,  r'$\text{argmax}\circ\mathcal{A}$', '--'),
    (influences_subb, lblue, r'$\text{argmax}\circ\widetilde{\mathcal{A}}_m$', '-'),
]:
    sorted_vals = np.sort(arr)
    n_pts = len(sorted_vals)
    x = np.concatenate([[0], sorted_vals])
    y = np.concatenate([[0], np.arange(1, n_pts+1) / n_pts])
    ax.step(x, y, where='post', color=c, label=lab, linestyle=ls, linewidth=1.5)

ax.set_xlabel(r'$\psi := \mathbb{E}_X \left\|\hat{p}(X) - \hat{p}^{-i}(X)\right\|$')
ax.set_ylabel(r'$\frac{1}{N}\sum_{i \in [N]} \mathbf{1}\{\psi_i \leq \psi\}$')
ax.legend(loc='lower right')
ax.spines[['bottom', 'left']].set_visible(True)
ax.spines[['top', 'right']].set_visible(True)
ax.set_ylim([0, 1])
ax.set_xlim(left=0)
ax.legend(loc='lower right', fontsize=10)

plt.savefig('euclidean_plot.pdf')
plt.show()

### Done ###