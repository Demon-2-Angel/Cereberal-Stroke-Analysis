#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\aniru\Machine Learning\Datasets\Cereberal Stroke\Cereberal_Dataset.csv")
df.head(10)


# In[3]:


df.describe()


# In[4]:


df.isnull().sum()


# BMI and Smoking Stroke has a lot of NULL| values

# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df['stroke'].value_counts()


# Very imbalace 0,1 parameters

# In[8]:


sns.countplot(x='stroke', data=df)
plt.title("Imbalance data")
plt.show()


# ## One Hot Encoding 

# Converting the categorical values, indicator variables

# In[9]:


df.columns


# In[10]:


df = pd.get_dummies(df,columns=['gender','ever_married','work_type','Residence_type','smoking_status'])
df.head(4)


# ## Missing Value Handelling

# In[11]:


from sklearn.impute import KNNImputer

imputer = KNNImputer(missing_values=np.nan)
tab = imputer.fit_transform(df)
df_new = pd.DataFrame(tab, columns=df.columns)
df_new.head(10)


# In[12]:


df_new.shape


# The Columns has been transformed from 12 to 22

# In[13]:


df_new.isnull().sum()


# In[14]:


df_new.dtypes


# ## Machine Learning Model

# In[15]:


X = df_new.drop('stroke',axis=1)
y = df_new['stroke']


# In[16]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=42)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report


# In[18]:


knn = KNeighborsClassifier()
nb = GaussianNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

models = [knn, nb, dt, rf]

for model in models:
    print("MODEL NAME: ", model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred))


# The models accurately predicts the 0 but not 1. The precision is close to zero in all the above cases, which means the model failed to predict the cases where chances for cerebral stroke was actually present. So the above models are useless.

# ## OverSampling
# 

# In[19]:


from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=1)
X_os, y_os = os.fit_resample(X,y)

X_train,X_test,y_train,y_test = train_test_split(X_os,y_os,test_size=0.3,random_state=1)

for model in models:
    print("MODEL NAME:",model)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test,y_pred))


# After applying SMOTE technique, the precision increased. For KNN it is 0.88, for Decision Tree it is 0.97 and RandomForest it is 0.95. So oversampling resulted in a better model that is capable of identifying the cases positive for stroke.

# ## UnderSampling

# In[24]:


from imblearn.under_sampling import RandomUnderSampler

us= RandomUnderSampler(random_state=1)
X_us, y_us = us.fit_resample(X,y)

X_train,X_test,y_train,y_test = train_test_split(X_us,y_us,test_size=0.3,random_state=1)

for model in models:
    print("MODEL NAME:",model)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test,y_pred))


# ### Combining Oversampling and Undersampling 

# SMOTEEN combines SMOTE and Edited Nearest Neighbours(ENN). SMOTEEN performs upsampling and downsampling at the same time.

# In[25]:


from imblearn.combine import SMOTEENN

sample = SMOTEENN()
X_over,y_over = sample.fit_resample(X,y)

X_train,X_test,y_train,y_test = train_test_split(X_over,y_over,test_size=0.3,random_state=1)


# In[27]:


for model in models:
    print("MODEL NAME:",model)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test,y_pred))


# <b>End Note</b> : For the Cerebral Stroke Imbalanced data we could make a better model using resampling techniques.

# In[ ]:




