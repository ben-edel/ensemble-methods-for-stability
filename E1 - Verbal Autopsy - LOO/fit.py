
import numpy as np
import pandas as pd
import pickle as pkl
import os, sys
import re
from datetime import datetime

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
model = sys.argv[3]
run_path = os.path.join(run_type, run_id)
if not os.path.exists(run_path): 
    os.makedirs(run_path)

if run_type == 'full':
    np.random.seed(0)
else:
    np.random.seed(int(run_id))

n = len(training_set)
m = int(n/2)

if run_type=='subbagging':
    inds = np.random.choice(n, m, replace=False)
    training_set = training_set.iloc[inds]
elif run_type=='loo':
    drop_ind = int(sys.argv[4])
    inds = np.delete(np.arange(n), drop_ind)
    training_set = training_set.iloc[inds]
else: 
    inds = np.arange(n)

with open(os.path.join(run_path, 'indices.pkl'), 'wb') as f:
    pkl.dump(inds, f)

# FIT MODEL

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = training_set[feature_cols]
X_test = validation_set[feature_cols]
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = training_set[target_col]
y_test = validation_set[target_col]

if model == 'mlp':
    # MLP
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(16,),
        alpha=0.01,
        early_stopping=True,
        n_iter_no_change=3,
        learning_rate_init=0.01,
        batch_size=256,
        random_state=42,
        max_iter=30,
        solver='adam'
    )
    mlp.fit(X_train_scaled, y_train)
    w = mlp.predict_proba(X_test_scaled)

elif model == 'gradient_boosting':
    # Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=1
    )
    gb.fit(X_train, y_train)
    w = gb.predict_proba(X_test)

elif model == 'logistic_regression':
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(
        max_iter=500,
        C=0.01, 
        penalty='l2',
        solver='saga',
        tol=1e-3,
        random_state=0,
        n_jobs=-1
    )
    lr.fit(X_train_scaled, y_train)
    w = lr.predict_proba(X_test_scaled)

elif model == 'lda':
    # LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_train_scaled, y_train)
    w = lda.predict_proba(X_test_scaled)

elif model == 'naive_bayes':
    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    w = nb.predict_proba(X_test)

elif model == 'knn':
    # K-NN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    )
    knn.fit(X_train_scaled, y_train)
    w = knn.predict_proba(X_test_scaled)

elif model == 'decision_tree':
    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(
        random_state=0,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features=None
    )
    dt.fit(X_train, y_train)
    w = dt.predict_proba(X_test)

elif model == 'reg_decision_tree':
    # Regularized Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(
        random_state=0,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10
    )
    dt.fit(X_train, y_train)
    w = dt.predict_proba(X_test)

else:
    raise ValueError(f"Unknown model: {model}")

# Ensure probabilities sum to 1
w = w/w.sum(1)[:, np.newaxis]

with open(os.path.join(run_path, 'weights.pkl'), 'wb') as f:
    pkl.dump(w, f)