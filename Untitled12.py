#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[22]:


data=pd.read_excel('V:/intern.xlsx')


# In[23]:


data


# In[24]:


data.info()


# In[25]:


data.plot(kind="scatter",x="Hours",y="Scores")


# In[30]:


Hr=pd.DataFrame(data["Hours"])
Sc=pd.DataFrame(data['Scores'])


# In[31]:


lm=linear_model.LinearRegression()
model=lm.fit(Hr,Sc)


# In[46]:


b0=model.intercept_
b0


# In[47]:


b1=model.coef_
b1


# In[48]:


r_sq=model.score(Hr,Sc)
r_sq


# In[49]:


#prediction
Sc_predict=b0+b1*9.25
Sc_predict


# 

# In[ ]:





# In[ ]:




