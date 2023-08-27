#!/usr/bin/env python
# coding: utf-8

# In[137]:


#KÜTÜPHANE İMPORTLARI VE VERİ OKUMA İŞLEMLERİ

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import xgboost as xgb


# In[3]:


df = pd.read_csv("Iris.csv")


# In[4]:


df.head()


# In[8]:


df.info()


# In[10]:


df.describe().T


# In[11]:


df.groupby("Species").agg(["min", "max", "std", "mean"])


# In[13]:


df.isna().sum()


# In[14]:


#VERİNİN GÖRSELLEŞTİRİLMESİ

df.head()


# In[17]:


sns.scatterplot(data=df, x="Id", y="SepalLengthCm", hue="Species" )


# In[18]:


sns.scatterplot(data=df, x="Id", y="PetalLengthCm", hue="Species" )


# In[20]:


for col in df.columns[1:-1]:
    sns.scatterplot(data=df, x="Id", y=col, hue="Species" )
    plt.show()


# In[42]:


#AYKIRI DEĞER İŞLEMLLERİ
#IQR YÖNTEMİ İLE AYKIRI DEĞER İŞLEMLERİ

def outlier_thresholds(dataframe,col_name,  q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3= dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * interquantile_range
    up_limit = quartile3 + 1.5 * interquantile_range
    return low_limit, up_limit


# In[43]:


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99 )
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False
    


# In[44]:


num_cols = [ col for col in df.columns if df[col].dtypes != "O"]
num_cols = [col for col in num_cols if col not in 'Id' ]
num_cols


# In[45]:


for col in num_cols:
    print(col, check_outlier(df, col))
    
 ## verisetinde herhangi bir aykırı değere rastlanmamıştır.


# In[53]:


#ENCODING İŞLEMLERİ

le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])


# In[55]:


df.head()
df["Species"].value_counts()


# In[56]:


#VERİ KONTROL İŞLEMLERİ

df.isnull().sum()


# In[57]:


df.dtypes


# In[123]:


df.head()


# In[122]:


df["Species"].value_counts()


# In[125]:


#MODELLEME İŞLEMLERİ

X_train, X_test, y_train, y_test = train_test_split(df.iloc[: , :-1], df.iloc[:, -1], test_size=0.2)


# In[127]:


xgb_cls = xgb.XGBClassifier(objective="multiclass:softmax", num_class=3)


# In[129]:


xgb_cls.fit(X_train, y_train)


# In[130]:


preds = xgb_cls.predict(X_test)


# In[132]:


X_test


# In[133]:


preds


# In[135]:


np.array(y_test)


# In[138]:


#METRİKLERİN İNCELENMESİ

accuracy_score(y_test, preds)


# In[140]:


confusion_matrix(y_test, preds)


# In[ ]:




