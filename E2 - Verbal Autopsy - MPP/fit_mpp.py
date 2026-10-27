
import numpy as np
import pandas as pd
import pickle as pkl
import random
import re
import os, sys
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

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

run_type = sys.argv[1]
run_id = sys.argv[2]
model_class = sys.argv[3]
perturbation_scheme = sys.argv[4]
run_path = os.path.join(run_type, run_id)
if not os.path.exists(run_path): 
    os.makedirs(run_path)

if run_type == 'full':
    np.random.seed(0)
else:
    np.random.seed(int(run_id))

n = len(training_set)
m = int(n/2)

if run_type == 'perturbed':
    if perturbation_scheme == 'bootstrap':
        original_training_set = training_set.copy()
        inds = np.random.choice(n, n, replace=True)
        training_set = original_training_set.iloc[inds].reset_index(drop=True)
        while training_set[target_col].nunique() != K:
            inds = np.random.choice(n, n, replace=True)
            training_set = original_training_set.iloc[inds].reset_index(drop=True)
    elif perturbation_scheme == 'leave_k_out':
        original_training_set = training_set.copy()
        k = 1000  # number of points to leave out
        inds = np.random.choice(n, n - k, replace=False)
        training_set = original_training_set.iloc[inds].reset_index(drop=True)
        while training_set[target_col].nunique() != K:
            inds = np.random.choice(n, n - k, replace=False)
            training_set = original_training_set.iloc[inds].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown perturbation scheme: {perturbation_scheme}")
else:
    inds = np.arange(n)

### Redefine training and test data ###
n = len(training_set)
m = int(n/2)
X_train = training_set.drop(target_col, axis=1)
y_train = training_set[target_col]
X_test = validation_set.drop(target_col, axis=1)
y_test = validation_set[target_col]

# FIT MODEL

# Determine model to use based on command line argument
if model_class == 'lda':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

elif model_class == 'naive_bayes':
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()

elif model_class == 'decision_tree':
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(
        random_state=0,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features=None
    )

elif model_class == 'reg_decision_tree':
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(
        random_state=0,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10
    )

else:
    raise ValueError(f"Unknown model class: {model_class}")     

# Fit model
if model_class == 'lda':
    model.fit(X_train_scaled, y_train)
    w = model.predict_proba(X_test_scaled)
else:
    model.fit(X_train, y_train)
    w = model.predict_proba(X_test)

# Normalize
w = w / w.sum(1)[:, np.newaxis]

with open(os.path.join(run_path, 'indices.pkl'), 'wb') as f:
    pkl.dump(inds, f)

with open(os.path.join(run_path, 'weights.pkl'), 'wb') as f:
    pkl.dump(w, f)

### SUBBAGGING ###
for i in range(1,501):
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

    # Define sb-model
    if model_class == 'lda':
        sb_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        sb_scaler = StandardScaler()
        sb_X_train_scaled = sb_scaler.fit_transform(sb_X_train)
        sb_X_test_scaled = sb_scaler.transform(X_test)

    elif model_class == 'naive_bayes':
        sb_model = GaussianNB()

    elif model_class == 'decision_tree':
        sb_model = DecisionTreeClassifier(
            random_state=0,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features=None
        )

    elif model_class == 'reg_decision_tree':
        sb_model = DecisionTreeClassifier(
            random_state=0,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10
        )

    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    try:
        if model_class == 'lda':
            sb_model.fit(sb_X_train_scaled, sb_y_train)
            sb_w = sb_model.predict_proba(sb_X_test_scaled)
        else:
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
        print(f"Subbagging {run_id} - {i}: Failed")
        continue

