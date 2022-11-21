#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# In[15]:


col_names=['x1','x2','result']
ds=pd.read_csv("std.csv",names=col_names)


# In[16]:


print(ds)
x = ds.iloc[:,[0,1]].values
y = ds.iloc[:,2].values


# In[18]:


xp=preprocessing.scale(x)
kf=KFold(n_splits=5)


# In[22]:


for train_index,test_index in kf.split(xp):
    xtrain,xtest,ytrain,ytest=train_test_split(xp,y,test_size=0.20,random_state=0)
    x1=xtrain[:,0]
    x2=xtrain[:,1]
    b0=0.0
    b1=0.0
    b2=0.0
    epoch=10000
    alpha=0.001
    while(epoch>0):
        for i in range(len(xtrain)):
            prediction=1/(1+np.exp(-(b0+b1*x1[i]+b2*x2[i])))
            b0=b0+alpha*(ytrain[i]-prediction)*prediction*(1-prediction)*1.0
            b1=b1+alpha*(ytrain[i]-prediction)*prediction*(1-prediction)*x1[i]
            b2=b2+alpha*(ytrain[i]-prediction)*prediction*(1-prediction)*x2[i]
        epoch=epoch-1
        print(b0)
        print(b1)
        print(b2)
            


# In[23]:


final_prediction=[]
x3=xtest[:,0]
x4=xtest[:,1]
print(ytest)


# In[24]:


y_pred=[0]*len(xtest)


# In[25]:


for i in range(len(xtest)):
    y_pred[i]=np.round(1/(1+np.exp(-(b0+b1*x3[i]+b2*x4[i]))))


# In[26]:


final_prediction.append(np.ceil(y_pred[i]))
print(final_prediction)


# In[27]:


from sklearn.metrics import accuracy_score
print("Accuracy",accuracy_score(ytest,y_pred))


# In[ ]:




