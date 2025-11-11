#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Experiment performed by:
## Aman Tyagi 23/CS/046


# ### Pre-Processing (Dataset and Setup)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Mall_Customers.xls')


# In[3]:


df.head()


# In[23]:


print(df.shape)


# In[24]:


print(df.columns)


# In[25]:


print(df.info())


# In[26]:


print(df.describe())


# In[4]:


df.isnull().sum()


# In[5]:


numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for col in numeric_cols:
  plt.figure(figsize=(6, 4))
  sns.boxplot(x=df[col])
  plt.title(f'boxplot of {col}')
  plt.show()


# In[6]:


## encoding the features
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df.head()


# In[28]:


from sklearn.preprocessing import MinMaxScaler

# Separate identifier column
customer_ids = df['CustomerID'].copy()

# Select only numerical features (in case there are non-numeric columns)
features = df.drop(columns=['CustomerID']).select_dtypes(include=['number']).copy()

# Initialize and apply MinMaxScaler
scaler = MinMaxScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(features),
    columns=features.columns,
    index=features.index
)

# Combine the scaled features back with the CustomerID column
df_scaled = pd.concat([customer_ids, features_scaled], axis=1)


# In[29]:


df_scaled.head()


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define the numeric columns to visualize
cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Set a consistent plotting style
sns.set(style="whitegrid")

# Plot distributions for each numeric feature
for col in cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=20, color='skyblue')
    plt.title(f"Distribution of {col}", fontsize=13, weight='bold')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


# ## Implementing K-means from scratch

# In[10]:


## initialisation

def initialize_centroids(X, k, random_state=None):
  if random_state is not None:
    np.random.seed(random_state)

  n_samples = X.shape[0]
  random_indices = np.random.choice(n_samples, size =k, replace=False)

  centroids = X[random_indices]

  return centroids


# In[11]:


def initialise_centroids_kmeanspp(X, k, random_state=None):
  if random_state is not None:
    np.random.seed(random_state)

  n_samples, n_features = X.shape

  first_centroid_idx = np.random.randint(n_samples)
  centroids = X[first_centroid_idx]

  for _ in range (1, k):
    distances = np.array([
        min(np.linalg.norm(x -c)**2 for c in centroids) for x in X
    ])

    probabilities = distances/ distances.sum()

    next_centroid_idx = np.random.choice(n_samples, p=probabilities)
    centroids.append(X[next_centroid_idx])

  return np.array(centroids)


# In[12]:


## assignment

def assign_clusters(X, centroids):
  labels = []

  for x in X:
    distances = []

    for c in centroids:
      distance = np.linalg.norm(x - c)
      distances.append(distance)

    label = np.argmin(distances)
    labels.append(label)

  return np.array(labels)


# In[13]:


def update_centroids(X, labels, k):
  new_centroids = []

  for i in range(k):
    points_in_cluster = X[labels == i]

    if len(points_in_cluster) > 0 :
      new_centroid = points_in_cluster.mean(axis=0)
      new_centroids.append(new_centroid)
    else:
      new_centroid = np.zeros(X.shape[1])
      new_centroids.append(new_centroid)

  return np.array(new_centroids)


# In[14]:


## main algorithm loop

def kmeans(X, k, max_iters=100, tol=1e-4, random_state=None):
  if random_state is not None:
    np.random.seed(random_state)

  n_samples, n_features = X.shape
  centroids = initialize_centroids(X, k, random_state)

  for i in range(max_iters):
      labels = assign_clusters(X, centroids)

      new_centroids = update_centroids(X, labels, k)

      shift = np.linalg.norm(new_centroids - centroids)

      if shift < tol:
          break

      centroids = new_centroids


  distances = []
  for idx, x in enumerate(X):
      c = centroids[labels[idx]]
      distances.append(np.linalg.norm(x - c) ** 2)
  inertia = np.sum(distances)

  return centroids, labels, i + 1, inertia


# In[15]:


## elbow method and silhouette score
## just importing silhouetter score from sklearn
from sklearn.metrics import silhouette_score

def evaluate_k_range(X, k_min=2, k_max = 10, random_state=None):
  intertias = []
  silhouettes = []

  for k in range(k_min, k_max+1):
    centroids, labels, iter, inertia = kmeans(X, k, random_state=random_state)
    intertias.append(inertia)

    if len(set(labels)) > 1:
      sil = silhouette_score(X, labels)
    else:
      sil = np.nan
    silhouettes.append(sil)

    print(f"k={k}: inertia={inertia:.3f}, silhouetter={sil:.3f}")


  results = pd.DataFrame({
      "k": range(k_min, k_max+1),
      "inertia": intertias,
      "silhouette": silhouettes
  })

  fig, ax1 = plt.subplots(figsize=(8,5))

  color = 'tab:blue'
  ax1.set_xlabel("Number of Clusters (k)")
  ax1.set_ylabel("Inertia", color=color)
  ax1.plot(results["k"], results["inertia"], marker='o', color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.set_title("Elbow Method and Silhouette Score")

  ax2 = ax1.twinx()
  color = 'tab:orange'
  ax2.set_ylabel("Silhouette Score", color=color)
  ax2.plot(results["k"], results["silhouette"], marker='s', linestyle='--', color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  plt.show()

  return results


# In[31]:


## applying the algorithmm
X = df_scaled.values
results = evaluate_k_range(X, k_min=2, k_max=10, random_state=43)

print(results)


# In[32]:


chosen_k = 3

centroids, labels, n_iter, inertia = kmeans(X, k=chosen_k,max_iters=100, tol=0.0001, random_state=42)

print("Final centroids:\n", centroids)
print("Inertia:", inertia)
print("Iterations:", n_iter)


# In[33]:


df_scaled["cluster"] = labels


# In[34]:


cluster_means = df_scaled.groupby("cluster").mean()
print(cluster_means)


# Cluster 0: Average-age, middle-income customers with moderate spending behavior.
# 
# Cluster 1: Younger, low-income customers who spend moderately
# 
# Cluster 2: Older, mid-income customers who spend consistently
# 
# Cluster 3: Younger, high-income customers with average spending levels

# In[35]:


plt.figure(figsize=(7,6))
plt.scatter(df_scaled.iloc[:, 2], df_scaled.iloc[:, 3], c=labels, cmap='tab10')
plt.scatter(centroids[:, 2], centroids[:, 3], c='black', marker='X', s=200)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.title('Visualisation')
plt.show()


# In[21]:


from sklearn.cluster import KMeans


# Run sklearn KMeans
sk_kmeans = KMeans(
    n_clusters=chosen_k,
    max_iter=100,
    tol=1e-4,
    random_state=42
)

sk_kmeans.fit(X)

sk_centroids = sk_kmeans.cluster_centers_
sk_inertia = sk_kmeans.inertia_
sk_iters = sk_kmeans.n_iter_

print("scikit-learn KMeans Results:")
print(f"Final Inertia: {sk_inertia:.4f}")
print(f"Iterations until convergence: {sk_iters}")
print(f"Centroids:\n{sk_centroids}")


# In[22]:


centroids_custom, labels_custom, n_iter_custom, inertia_custom = kmeans(
    X, k=chosen_k, max_iters=100, tol=1e-4, random_state=42
)

sk_kmeans = KMeans(n_clusters=chosen_k, max_iter=100, tol=1e-4, random_state=42)
sk_kmeans.fit(X)
labels_sk = sk_kmeans.labels_
centroids_sk = sk_kmeans.cluster_centers_

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_idx, y_idx = 2, 3

# Custom K-Means plot
axes[0].scatter(X[:, x_idx], X[:, y_idx], c=labels_custom, cmap='viridis', s=30)
axes[0].scatter(centroids_custom[:, x_idx], centroids_custom[:, y_idx], c='black', marker='X', s=200)
axes[0].set_title("Custom K-Means Clusters")
axes[0].set_xlabel(df_scaled.columns[x_idx])
axes[0].set_ylabel(df_scaled.columns[y_idx])

# scikit-learn K-Means plot
axes[1].scatter(X[:, x_idx], X[:, y_idx], c=labels_sk, cmap='viridis', s=30)
axes[1].scatter(centroids_sk[:, x_idx], centroids_sk[:, y_idx], c='black', marker='X', s=200)
axes[1].set_title("scikit-learn KMeans Clusters")
axes[1].set_xlabel(df_scaled.columns[x_idx])
axes[1].set_ylabel(df_scaled.columns[y_idx])

plt.tight_layout()
plt.show()

