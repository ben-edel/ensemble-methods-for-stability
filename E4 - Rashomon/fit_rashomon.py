
import numpy as np
import pandas as pd
import pickle as pkl
import random
import re
import os, sys
from tqdm import tqdm
from pysortd import SORTDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

def rashomon_ensemble_predict_proba(model, X_data, n_classes):
    X_bin = model.binarizer_.transform(X_data)
    X_np = np.ascontiguousarray(X_bin, dtype=np.intc)
    
    n_samples = X_np.shape[0]
    n_trees = model.rashomon_set_size
    preds = np.zeros((n_samples, n_trees), dtype=int)
    
    for i in range(n_trees):
        tree = model.get_tree_n(i)
        for j in range(n_samples):
            node = tree
            while True:
                if node.is_leaf_node():
                    preds[j, i] = node.label
                    break
                if X_np[j, node.feature] == 0:
                    node = node.left_child
                else:
                    node = node.right_child
    
    probs = np.zeros((n_samples, n_classes))
    for c in range(n_classes):
        probs[:, c] = (preds == c).mean(axis=1)
    
    return probs

### FETCH PARAMETERS ###

run_type = sys.argv[1]
run_id = sys.argv[2]
perturbation_scheme = sys.argv[3]
dataset = sys.argv[4]
run_path = os.path.join(run_type, run_id)
if not os.path.exists(run_path): 
    os.makedirs(run_path)

### LOAD DATA ###

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

if run_type == 'full':
    np.random.seed(0)
else:
    np.random.seed(int(run_id))

n = len(training_set)
m = int(n/2)

if run_type=='perturbed':
    if perturbation_scheme == "loo":
        # Jackknife resample
        drop_ind = np.random.randint(0, len(training_set))
        inds = np.delete(np.arange(n), drop_ind)
        training_set = training_set.iloc[inds].reset_index(drop=True)
    elif perturbation_scheme == 'leave_k_out':
        # Leave-k-out resample
        original_training_set = training_set.copy()
        if dataset == "synthetic":
            k = 25  # number of points to leave out
        else:
            k = 1000
        inds = np.random.choice(n, n - k, replace=False)
        training_set = original_training_set.iloc[inds].reset_index(drop=True)
        while training_set[target_col].nunique() != K:
            inds = np.random.choice(n, n - k, replace=False)
            training_set = original_training_set.iloc[inds].reset_index(drop=True)
    elif perturbation_scheme == "bootstrap":
        # Bootstrap resample
        original_training_set = training_set.copy()
        inds = np.random.choice(n, n, replace=True)
        training_set = original_training_set.iloc[inds].reset_index(drop=True)
        while training_set[target_col].nunique() != K:
            inds = np.random.choice(n, n, replace=True)
            training_set = original_training_set.iloc[inds].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown perturbation scheme: {perturbation_scheme}")
else:
    inds = np.arange(n)

### Redefine training and test data ###

n = len(training_set)
m = int(n/2)
n_classes = training_set[target_col].nunique()
X_train = training_set.drop(target_col, axis=1)
y_train = training_set[target_col]
X_test = validation_set.drop(target_col, axis=1)
y_test = validation_set[target_col]

### SORTeD Parameters ###
if dataset == "synthetic":
    cost_complexity = 0.01
elif dataset == "heloc":
    cost_complexity = 0.005
else:
    raise ValueError(f"Unknown dataset: {dataset}")

### SINGLE MODEL ###

model = SORTDClassifier("cost-complex-accuracy", verbose=False, cost_complexity=cost_complexity, 
                        use_rashomon_multiplier=False, max_num_trees=5000)

model.fit(X_train, y_train)

# Get probs
w = model.predict_proba(X_test)

# Normalize
w = w / w.sum(1)[:, np.newaxis]

with open(os.path.join(run_path, 'indices.pkl'), 'wb') as f:
    pkl.dump(inds, f)

with open(os.path.join(run_path, 'weights.pkl'), 'wb') as f:
    pkl.dump(w, f)

### SUBBAGGING ###
max_num_trees = 1
for i in range(1,250):
    sb_training_set = training_set.copy()
    sb_inds = np.random.choice(n, m, replace=False)
    sb_training_set = sb_training_set.iloc[sb_inds].reset_index(drop=True)

    class_counts = sb_training_set[target_col].value_counts()
    if class_counts.min() < 2:
        print(f"Subbagging {run_id} - {i}: Insufficient minority class, skipping")
        continue

    sb_X_train = sb_training_set.drop(target_col, axis=1)
    if (sb_X_train.nunique() == 1).all():
        print(f"Subbagging {run_id} - {i}: All features constant, skipping")
        continue

    sb_y_train = sb_training_set[target_col]

    sb_model = SORTDClassifier("cost-complex-accuracy", verbose=False, cost_complexity=cost_complexity, 
                               use_rashomon_multiplier=False, max_num_trees=1)
    try:
        sb_model.fit(sb_X_train, sb_y_train)
        sb_w = sb_model.predict_proba(X_test)
        sb_w = sb_w / sb_w.sum(1)[:, np.newaxis]

        bs_run_id = '0' if run_type=='full' else run_id
        bs_path = os.path.join('subbagging', bs_run_id, str(i))

        if not os.path.exists(bs_path):
            os.makedirs(bs_path)

        sb_inds_original = inds[sb_inds]
        with open(os.path.join(bs_path, 'indices.pkl'), 'wb') as f:
            pkl.dump(sb_inds_original, f)

        with open(os.path.join(bs_path, 'weights.pkl'), 'wb') as f:
            pkl.dump(sb_w, f)
    except:
        print(f"Subbagging {i}: Failed")
        continue

### RASHOMON ###

rashomon_set_size = model.rashomon_set_size

# Get rashomon ensemble
X_bin = model.binarizer_.transform(X_test)
X_np = np.ascontiguousarray(X_bin, dtype=np.intc)

n_samples = X_np.shape[0]
n_trees = model.rashomon_set_size
all_preds = np.empty((n_samples, n_trees), dtype=np.int8)

for i in range(n_trees):
    tree = model.get_tree_n(i)
    mask = np.ones(n_samples, dtype=bool)
    preds_i = np.zeros(n_samples, dtype=np.float64)
    
    def traverse(node, mask):
        if not np.any(mask): return
        try: is_leaf = node.is_leaf_node()
        except TypeError: is_leaf = node.is_leaf_node
        if is_leaf:
            preds_i[mask] = node.label
        else:
            traverse(node.left_child,  mask & (X_np[:, node.feature] == 0))
            traverse(node.right_child, mask & (X_np[:, node.feature] == 1))
    
    traverse(tree, mask)
    all_preds[:, i] = preds_i

# Prune trees by identical predictions
pruned_np = np.unique(all_preds.T, axis=0).T
'''
seen = set()
kept_cols = []
for i in range(n_trees):
    col = tuple(all_preds[:, i])
    if col not in seen:
        seen.add(col)
        kept_cols.append(i)
    if len(kept_cols) == 100:
        break
pruned_np = all_preds[:, kept_cols]
'''

from sklearn.cluster import KMeans

n_unique = pruned_np.shape[1]

if dataset == "synthetic":
    n_clusters = min(10, n_unique)
else:
    n_clusters = min(50, n_unique)

# Cluster on transposed: each row is a tree's prediction vector
tree_vecs = pruned_np.T  # shape: (n_unique, n_samples)

if n_clusters < n_unique:
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = km.fit_predict(tree_vecs)
    
    # For each cluster, take majority vote prediction per sample
    clustered_preds = np.zeros((n_samples, n_clusters), dtype=np.int8)
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 1:
            clustered_preds[:, c] = tree_vecs[mask][0]
        else:
            # Majority vote within cluster
            clustered_preds[:, c] = (tree_vecs[mask].mean(axis=0) >= 0.5).astype(np.int8)
else:
    # No clustering needed — already at or below target
    clustered_preds = pruned_np

print(f" Pruning: {all_preds.shape[1]} -> {pruned_np.shape[1]} -> {clustered_preds.shape[1]}")

if run_type == 'full':
    n_clusters_actual = clustered_preds.shape[1]
    # Shape: (n_samples, n_clusters, n_clusters)
    disagree_matrix = (clustered_preds[:, :, None] != clustered_preds[:, None, :]).mean(axis=0)
    # Shape: (n_clusters, n_clusters)
    i_upper, j_upper = np.triu_indices(n_clusters_actual, k=1)
    avg_pairwise_disagreement = disagree_matrix[i_upper, j_upper].mean()
    print(f"Average Pairwise Disagreement: {avg_pairwise_disagreement:.4f}")

probs = np.zeros((n_samples, K))
for c in range(K):
#    probs[:, c] = (all_preds == c).mean(axis=1)
#    probs[:, c] = (pruned_np == c).mean(axis=1)
    probs[:, c] = (clustered_preds == c).mean(axis=1)

# Get weights from averaged rashomon
w_rashomon = probs
w_rashomon = w_rashomon / w_rashomon.sum(1)[:, np.newaxis]

if run_type=='perturbed':
    rashomon_path = os.path.join('rashomon', run_id)
else:
    rashomon_path = os.path.join('rashomon', '0')

if not os.path.exists(rashomon_path): 
    os.makedirs(rashomon_path)

with open(os.path.join(rashomon_path, 'indices.pkl'), 'wb') as f:
    pkl.dump(inds, f)

with open(os.path.join(rashomon_path, 'weights.pkl'), 'wb') as f:
    pkl.dump(w_rashomon, f)
