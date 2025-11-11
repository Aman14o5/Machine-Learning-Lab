#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# 
# 
# 
# load a CSV dataset from a local file using `pandas.read_csv,,,
# use basic Pandas functions for Exploratory Data Analysis-EDA
# describe and discriminate between basic data types such as categorical, quantitative, continuous, discrete, ordinal, nominal and identifier

# # [Objective 01] LOAD THE DATASET AND EXPLORE

# ## Overview
# 
# 
# 
# Steps for loading a dataset:
# 
# 1) Learn as much as you can about the dataset:
#  - Number of rows
#  - Number of columns
#  - Column headers (Is there a "data dictionary"?)
#  - Is there missing data?
#  - Open the raw file and look at it - it may not be formatted the way you expect.
# 
# 2) Try loading the dataset using `pandas.read_csv()` and if things aren't acting the way that you expect, investigate until you can get it loading correctly.
# 
# 3) Keep in mind that functions like `pandas.read_csv()` have a lot of optional parameters that might help us change the way that data is read in. If you get stuck, google, read the documentation, and try things out.
# 
# 4) You might need to type out column headers by hand if they are not provided in a neat format in the original dataset. It can be a drag.

# ## Follow Along

# ### Learn about the dataset and look at the raw file.

# In[51]:


# Open and read raw CSV
with open("./california_housing_test.csv", "r") as f:
    raw_data = f.read()

print(raw_data[:1000])#as the raw data is very large we print only first 1000 


# ### Attempt to load it.

# In[3]:


import pandas as pd


# In[5]:


df = pd.read_csv('./california_housing_test.csv')


# In[6]:


#Print the first 5 rows:
df.head()


# In[7]:


#Print the last 5 rows:
df.tail()


# In[8]:


#Can you print the first 7 rows?
df.head(7)


# In[9]:


df.duplicated()
duplicates = df[df.duplicated()]
print(duplicates)


# In[10]:


# You can look at the documentation, but generally, I recommend Googling.


# In[12]:


# You can provide the column headers if not already present in the dataset and pass it into the read_csv as shown
# column_headers = []
# dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data",
#                     names = column_headers)

columns = [
    "Longitude",
    "Latitude",
    "Housing_median_age",
    "Total_rooms",
    "Total_bedrooms",
    "Population",
    "Households",
    "Median_income",
    "Median_house_value"
]
df = pd.read_csv('./california_housing_test.csv',names = columns)

df.head()


# In[13]:


#Print the dimensions
#dimension = rows x columns
df.shape


# In[14]:


# Print the header again
df.head()


# In[15]:


#printing random 10 rows from the dataset
df.sample(10)


# In[16]:


#names of the respective columns in the dataset
df.columns.tolist()


# In[17]:


df


# # [Objective 02](#basic-pandas-functions) - Use basic Pandas functions for Exploratory Data Analysis (EDA)

# ## Overview
# 
# > Exploratory Data Analysis (EDA) refers to the critical process of performing initial investigations on data so as to discover patterns, to spot anomalies, to test hypotheses and to check assumptions with the help of summary statistics and graphical representations
# 
# Exploratory Data Analysis is often the first thing that we'll do when starting out with a new dataset.

# In[18]:


#Learn more about the variables in the dataset using info function
df.info()


# #Learn more about each variable

# In[19]:


#Determine the data types
print(df.dtypes)


# In[20]:


# Summary Statistics - using (describe function)
# check if there are non-numeric column
df.describe()


# What is the problem here?

# ????

# In[21]:


# try to exclude non numeric value
# df.describe(include=['number'])


#all the datatypes are non-numeric i.e objects


# In[22]:


# include all
df.describe(include='all')


# In[23]:


# accesss a specific column of the dataframe
print(df['Median_house_value'])


# In[24]:


print(df['Households'])


# In[25]:


print(df['Longitude'].value_counts())


# In[26]:


#sort by values (any specific column) i.e Housing_median_age

sorted_values = df.sort_values(by='Housing_median_age')
print(sorted_values)


# In[27]:


print(df.sort_values(by='Households'))


# In[28]:


print(df.sort_values(by='Median_house_value'))


# In[29]:


# check for missing values
# the number of missing values in each column

print(df.isnull())
print(df.isnull().sum())


# In[30]:


# Total number of missing cells in the entire dataset
total_missing = df.isnull().sum().sum()
print('Total Missing',total_missing)


# In[31]:


df.head()


# In[32]:


# try dropping rows from the dataset inplace
#dropped the 5th row inplace
df.drop(6,inplace=True)


# In[33]:


df.head()


# In[34]:


# axis=1 to look through column headers and not row index
#Drop ID variable

df.drop('Latitude',axis=1,inplace=True)

#inplace
#drop(column_name, axis=1, inplace=True)# Columns gets removed


# In[35]:


df.head()


# Recap - what do each of these things do???
# 
# 
# - df.shape
# - df.head()
# - df.dtypes
# - df.describe()
#  - Numeric
#  - Non-Numeric
# - df['column'].value_counts()
# - df.isnull().sum()
# - df.isnull().sum().sum()
# - df.drop()

# # [Objective 03](#pandas-visualizations) Describe and discriminate between basic data types

# ## Overview
# 
# One of the cornerstones of Exploratory Data Analysis (EDA) is being able to identify variable types.  We will need different statistical methods to display and describe each of these different types of data.

# In[52]:


df = pd.read_csv('./tmdb_5000_movies.csv')


# In[53]:


df.head()


# In[54]:


print(df.dtypes)


# In[55]:


print(df.info())


# In[56]:


df.describe()


# In[57]:


# Numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

# Boolean columns
bool_cols = df.select_dtypes(include=['bool']).columns

# Date/Time columns (if any)
date_cols = df.select_dtypes(include=['datetime64']).columns


# In[58]:


print("Numeric:", num_cols)


# In[59]:


print("Categorical:", cat_cols)


# In[60]:


print("Boolean:", bool_cols)


# In[61]:


print("Date/Time:", date_cols)


# In[62]:


print(df[num_cols].describe())   # mean, std, min, max, quartiles
df[num_cols].hist(bins=20, figsize=(10,6))


# In[50]:


for col in cat_cols:
    print(df[col].value_counts())
    print(df[col].value_counts(normalize=True))  # proportions


