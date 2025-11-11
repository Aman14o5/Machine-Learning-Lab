#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Loading the Dataset

# In[2]:


df_train = pd.read_csv('./california_housing_train.csv')
df_test = pd.read_csv('./california_housing_test.csv')


# In[3]:


df_train.head()


# In[4]:


df_test.head()


# **TASKS**

# A. Data Preprocessing

# In[5]:


df_train.info()


# In[6]:


# shape of input data
print(df_train.shape)


# In[7]:


#total missing values per column
print(df_train.isnull().sum())


# In[8]:


#check for duplicated rows
print(df_train.duplicated().sum())


# In[9]:


X_train = np.array(df_train.iloc[:,:-1])
y_train = np.array(df_train.iloc[:,-1]).reshape(-1,1)

X_test = np.array(df_test.iloc[:,:-1])
y_test = np.array(df_test.iloc[:,-1]).reshape(-1,1)


# In[10]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[11]:


y_train


# Standardisation of input features
# Mathematically:
#  z = (x-
# mean of the feature)/
# standard deviation of the feature
# 

# In[12]:


def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # to avoid division by zero (if a feature has constant value)
    std[std == 0] = 1
    X_std = (X - mean) / std
    return X_std, mean, std


# In[13]:


Std_X_train,mean_train,std_train = standardize(X_train)
Std_X_test,mean_test,std_test = standardize(X_test)


# In[14]:


print(Std_X_train.shape)
print(Std_X_test.shape)


# In[15]:


print(mean_train)
print(mean_test)


# In[16]:


print(Std_X_test)


# In[17]:


print(Std_X_train)


# In[18]:


class NormalEquation:
    def fit(self, X, y):
        # Add bias column (intercept term)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal equation: θ = (XᵀX)⁻¹Xᵀy
        self.theta = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)

    def predict(self, X):
        # Add bias column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta


# In[19]:


ne = NormalEquation()


# In[20]:


ne.fit(Std_X_train,y_train)


# In[21]:


print("Learned Parameters: ",ne.theta)


# In[22]:


y_pred = ne.predict(Std_X_test)


# In[23]:


print("Predicted Values: ",y_pred)


# In[24]:


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)   # residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # total sum of squares
    return 1 - (ss_res / ss_tot)


# In[25]:


print(r2_score(y_test, y_pred))


# In[26]:


class GradientDescent:
    def __init__(self):
        pass

    def fit(self, x_train, y_train, epochs=10000, learning_rate=0.01):
        rows, cols = x_train.shape

        # Add bias term
        modified_x_train = np.c_[np.ones((rows, 1)), x_train]
        theta = np.zeros((cols + 1, 1))
        transpose_x = modified_x_train.T

        # Store training history
        loss_per_iterations = []
        weights = []
        val_loss = []

        for i in range(epochs):
            # Forward pass
            y_pred = modified_x_train @ theta

            # Gradient descent update
            gradient = (transpose_x @ (y_pred - y_train)) / rows
            theta -= learning_rate * gradient

            # Training loss
            train_loss = np.mean((y_pred - y_train) ** 2)
            loss_per_iterations.append(train_loss)

            # Store weights (excluding bias term for clarity)
            weights.append(theta[1:].flatten().copy())

            # Validation loss same as training loss (on training data itself)
            val_loss.append(train_loss)

        # Save attributes
        self.theta = theta
        self.loss_per_iterations = loss_per_iterations
        self.weights = weights
        self.val_loss = val_loss  # validation loss is training loss on same data

    def predict(self, x_test):
        modified_x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
        return modified_x_test @ self.theta


# In[27]:


gd = GradientDescent()


# In[28]:


gd.fit(Std_X_train,y_train,epochs=1000,learning_rate=0.01)


# In[29]:


y_pred = gd.predict(Std_X_test)


# In[30]:


print(y_pred)


# In[ ]:





# In[31]:


accuracy = r2_score(y_test, y_pred)
print("R² Accuracy:", accuracy)


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[34]:


model = LinearRegression()


# In[35]:


model.fit(Std_X_train,y_train)


# In[36]:


y_pred = model.predict(Std_X_test)


# In[37]:


from sklearn.metrics import mean_squared_error


# In[38]:


print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# **VISUALIZATIONS**

# In[39]:


curves = []
epochs = 1000

def run_experiment(x_train, y_train, learning_rates, label_prefix):
    for lr in learning_rates:
        gd = GradientDescent()
        gd.fit(x_train=x_train, y_train=y_train, epochs=epochs, learning_rate=lr)
        curves.append((f"{label_prefix}, lr={lr}", gd.loss_per_iterations))

# Five learning rates for unscaled data
unscaled_lrs = [1e-8, 1e-9, 1e-10]
run_experiment(X_train, y_train, unscaled_lrs, "unscaled")

# Five learning rates for scaled data
scaled_lrs = [0.0001, 0.00001, 0.0003, 0.001, 0.000003]
run_experiment(Std_X_train, y_train, scaled_lrs, "scaled")


# In[40]:


plt.figure(figsize=(10,7))
for label, loss in curves:
    plt.scatter(range(len(loss)), loss, label=label)

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Gradient Descent: Loss vs Iterations (Scaled vs Unscaled)")
plt.yscale("linear")
plt.legend()
plt.tight_layout()
plt.show()


# Visualization using
# **Gradient Descent**

# In[41]:


gd = GradientDescent()


# In[42]:


gd.fit(Std_X_train,y_train,epochs=500,learning_rate=0.01)


# In[43]:


print(len(gd.weights))


# In[44]:


print(len(gd.loss_per_iterations))


# In[ ]:





# In[45]:


import matplotlib.pyplot as plt

num_params = len(gd.weights[0])  # Number of parameters per epoch
color = 'blue'  # Choose a single color

for param_index in range(num_params):
    param_values = [w[param_index] for w in gd.weights]

    plt.figure()
    plt.plot(param_values, gd.loss_per_iterations, color=color, label=f"Param {param_index}")
    plt.title(f"Parameter {param_index} vs Loss")
    plt.xlabel(f"Parameter {param_index} value")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# In[46]:


final_weights = gd.weights[-1]
print("Final weights:", final_weights)


# ***VALIDATION LOSS***

# In[47]:


print(gd.val_loss)


# In[48]:


plt.plot(range(len(gd.val_loss)), gd.val_loss)
plt.xlabel("Iteration")
plt.ylabel("Validation Loss")
plt.title("Gradient Descent: Validation Loss vs Iterations")
plt.yscale("linear")
plt.tight_layout()
plt.show()


# Evaluation Metrics

# In[49]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[50]:


def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Metrics for {model_name}:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Coefficient of Determination (R2): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print()


# In[51]:


model = LinearRegression()
model.fit(Std_X_train,y_train)


# In[52]:


y_pred_normal = ne.predict(Std_X_test)
y_pred_gd = gd.predict(Std_X_test)
y_pred_sklearn = model.predict(Std_X_test)


# In[53]:


print_metrics(y_test, y_pred_normal, "Normal Equation")
print_metrics(y_test, y_pred_gd, "Gradient Descent")
print_metrics(y_test, y_pred_sklearn, "Scikit-Learn Model")


# In[ ]:





# In[ ]:




