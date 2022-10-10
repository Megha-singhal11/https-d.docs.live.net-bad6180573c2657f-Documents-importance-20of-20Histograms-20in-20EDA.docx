#!/usr/bin/env python
# coding: utf-8

# ##  importing important libraries
# #### 1. pandas 2. numpy 3. matplotlib 4. seaborn

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## reading the csv data file in pandas

# In[2]:


df = pd.read_csv(r"C:\Users\Hp\Downloads\archive (1)\IRIS.csv")


# ## first 5 data ebtries of the dataset

# In[3]:


df.head()


# ## information of the complete dataset

# In[4]:


df.info()


# ## small dataset only for understanding the concept

# In[5]:


df.shape


# ## segregating datatypes of the complete dataset 
#       ## as categorical and numerical data

# In[6]:


categorical_features = [fea for fea in df.columns if df[fea].dtypes == 'O']
print(f'we have {categorical_features} as our categorical feature')


# In[7]:


numerical_features = [fea for fea in df.columns if df[fea].dtypes != 'O']
print(f'we have {numerical_features} as our numerical feature')


# ## univariate analysis

# ### checking how categorical data is categorised under 3 species

# In[8]:


df['species'].value_counts()


# ### segregating or grouping same category of species together for analysis

# In[9]:


setosa = df.loc[df['species']== 'Iris-setosa']
versicolor = df.loc[df['species']== 'Iris-versicolor']
virginica = df.loc[df['species']== 'Iris-virginica']


# ### univariate means only one feature (can be numerical or categorical)
# ### analysis for univariate categorical species grouping them 
#     ## by flowers numerical feature (only one out of all)
#         ## able to see how species grouped under that numerical feature.

# ## ploting univariate graph via matplotlib

# In[10]:


plt.plot(setosa['sepal_length'], np.zeros_like(setosa['sepal_length']), 'o')
plt.plot(versicolor['sepal_length'], np.zeros_like(versicolor['sepal_length'])
         , 'o')
plt.plot(virginica['sepal_length'], np.zeros_like(virginica['sepal_length'])
         , 'o')
plt.xlabel('sepal_length')
plt.show()


# ## problem detect :
# ## the values are overlapping for sepal_length species for all species categories
#   ## can not see the data counts clearly
#     
#      ##### solution : use histograms as y axis are counts and
#                   ##### x axis is for one feature(sepal_length)

# ## Histogram

# ### histogram shows clear picture for dataset
#  ### how species under sepal_length is distribured
### clear classification along with counts of feature data.
# In[11]:


x = [setosa['sepal_length'], virginica['sepal_length']
     , versicolor['sepal_length']]
plt.hist(x, bins = 5)
plt.xlabel('sepal_length')
plt.ylabel('counts')
plt.show()


# In[ ]:




