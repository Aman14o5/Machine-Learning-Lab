#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Experiment performed by:
## Aman Tyagi 23/CS/046


# # Decision Tree Implementation from Scratch

# ## Importing Important Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ## Importing Dataset

# In[2]:


# get_ipython().run_line_magic('pip', 'install ucimlrepo')


# In[3]:


from ucimlrepo import fetch_ucirepo


# In[4]:


adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets


# ## Handling Missing Values

# In[5]:


print(X.isnull().sum())

X = X.dropna()
y = y.loc[X.index]


# ## Encode categorical columns

# In[6]:


encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])
    encoders[col] = le


# ## Splitting the Dataset

# In[7]:


# First split into train + temp (60/40)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Split temp into validation (20%) and test (20%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)


# ## Functions to Calculate Entropy and Gini

# In[9]:


# Entropy
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))  # avoid log(0)

# Gini
def gini(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


# ## Function to decide the Best Split

# In[10]:


def best_split(X, y, criterion="gini"):
    best_feature, best_thresh, best_gain = None, None, -1
    n_samples, n_features = X.shape
    base_impurity = gini(y) if criterion == "gini" else entropy(y)

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for thresh in thresholds:
            left_idx = X[:, feature] <= thresh
            right_idx = X[:, feature] > thresh
            if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                continue

            left_imp = gini(y[left_idx]) if criterion == "gini" else entropy(y[left_idx])
            right_imp = gini(y[right_idx]) if criterion == "gini" else entropy(y[right_idx])

            n_left, n_right = len(y[left_idx]), len(y[right_idx])
            weighted_impurity = (n_left * left_imp + n_right * right_imp) / n_samples
            gain = base_impurity - weighted_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_thresh = thresh

    return best_feature, best_thresh, best_gain


# ## Defining the Nodes of the Decision Tree

# In[11]:


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value   # prediction if leaf


# ## Function to Predict the Sample according to the given Tree

# In[12]:


def predict_sample(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_sample(node.left, x)
    else:
        return predict_sample(node.right, x)

def predict(node, X):
    return np.array([predict_sample(node, sample) for sample in X])


# ## Pre-Pruning

# In[13]:


def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2, min_impurity_decrease=1e-7, criterion="gini"):
    n_samples, n_features = X.shape

    # Leaf condition: pure node OR max_depth reached OR not enough samples
    if (len(np.unique(y)) == 1
        or (max_depth is not None and depth >= max_depth)
        or n_samples < min_samples_split):
        values, counts = np.unique(y, return_counts=True)
        return Node(value=values[np.argmax(counts)])

    # Find best split
    feature, threshold, gain = best_split(X, y, criterion)
    if gain < min_impurity_decrease:  # no significant improvement
        values, counts = np.unique(y, return_counts=True)
        return Node(value=values[np.argmax(counts)])

    # Split dataset
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold

    left_child = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth, min_samples_split, min_impurity_decrease, criterion)
    right_child = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth, min_samples_split, min_impurity_decrease, criterion)

    return Node(feature, threshold, left_child, right_child)


# ## Post-Pruning

# In[14]:


def majority_class(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]

def prune_tree(node, X_val, y_val):
    if node.value is not None:
        return node

    node.left = prune_tree(node.left, X_val, y_val)
    node.right = prune_tree(node.right, X_val, y_val)

    preds_before = predict(node, X_val)
    acc_before = (preds_before == y_val).mean()

    backup_left, backup_right = node.left, node.right

    node.left = node.right = None
    node.value = majority_class(y_val)

    preds_after = predict(node, X_val)
    acc_after = (preds_after == y_val).mean()

    if acc_after < acc_before:
        node.left, node.right = backup_left, backup_right
        node.value = None

    return node


# <!-- ###**Evaluation** -->

# ## Performing the Main Experiments

# In[15]:


def clean_labels(arr):
    return np.array([str(label).strip().replace(".", "") for label in arr])

def evaluate(y_true, y_pred):
    y_true = clean_labels(y_true)
    y_pred = clean_labels(y_pred)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=">50K"),
        "Recall": recall_score(y_true, y_pred, pos_label=">50K"),
        "F1": f1_score(y_true, y_pred, pos_label=">50K")
    }

def train_eval(max_depth=None, criterion="gini", min_samples_split=2, min_impurity_decrease=1e-7):
    tree = build_tree(
        X_train.values, y_train.values.ravel(),
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_impurity_decrease=min_impurity_decrease,
        criterion=criterion
    )
    yv = predict(tree, X_val.values)
    yt = predict(tree, X_test.values)
    return tree, evaluate(y_val.values.ravel(), yv), evaluate(y_test.values.ravel(), yt)


# ## Comparing both Gini and Entropy

# In[16]:


results = []
for crit in ["gini", "entropy"]:
    tree = build_tree(X_train.values, y_train.values.ravel(), max_depth=6, criterion=crit)
    preds_val = predict(tree, X_val.values)
    preds_test = predict(tree, X_test.values)
    results.append({
        "Criterion": crit,
        "Val_Accuracy": accuracy_score(y_val, preds_val),
        "Test_Accuracy": accuracy_score(y_test, preds_test)
    })

pd.DataFrame(results)


# ## Comparing different Depths

# In[17]:


depths = [2, 4, 6, None]
for d in depths:
    tree, val_metrics, test_metrics = train_eval(max_depth=d, criterion="gini")
    print(f"Depth={d}:  VAL {val_metrics}   TEST {test_metrics}")


# ## Showing Effect of Pruning

# In[18]:


# A) FULL TREE (no pre-pruning)
full_tree, val_full, test_full = train_eval(max_depth=None, criterion="gini")
results = {
    "FULL TREE": {"VAL": val_full, "TEST": test_full}
}

# B) PRE-PRUNED (fixed depth = 4 for example)
pre_tree, pre_val, pre_test = train_eval(max_depth=4, criterion="gini", min_samples_split=5)
results["PRE-PRUNED (depth=4)"] = {"VAL": pre_val, "TEST": pre_test}

# C) POST-PRUNED (reduced-error pruning)
pruned_tree = prune_tree(full_tree, X_val.values, y_val.values.ravel())
yva = predict(pruned_tree, X_val.values)
yta = predict(pruned_tree, X_test.values)
post_val = evaluate(y_val.values.ravel(), yva)
post_test = evaluate(y_test.values.ravel(), yta)
results["POST-PRUNED"] = {"VAL": post_val, "TEST": post_test}

# --- Print Summary Neatly ---
print("=== Decision Tree Comparison (Validation vs Test) ===\n")
for k, v in results.items():
    print(f"{k}:")
    print(f"   VAL  → {v['VAL']}")
    print(f"   TEST → {v['TEST']}\n")


# ## Identifying the most important features

# In[20]:


feat_names = np.array(list(X_train.columns))

def root_feature_name(tree):
    return feat_names[tree.feature] if tree.feature is not None else None

def collect_split_counts(node, counts):
    if node is None or node.value is not None:
        return
    counts[node.feature] = counts.get(node.feature, 0) + 1
    collect_split_counts(node.left, counts)
    collect_split_counts(node.right, counts)

def top_k_features(node, k=5):
    counts = {}
    collect_split_counts(node, counts)
    order = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(feat_names[i], c) for i, c in order]

print("Root feature:", root_feature_name(full_tree))
print("Top split features:", top_k_features(full_tree, k=10))

