#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
from apyori import apriori
import numpy as np 
from matplotlib import pyplot as plt 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# In[2]:


store_data = pd.read_csv("examScore.csv", header=None)
display(store_data.head())


# In[3]:


import numpy as np
from matplotlib import pyplot as plt


# In[4]:


raw_data=pd.read_csv("examScore.csv")


# In[5]:


raw_data.head()


# In[6]:


raw_data.shape


# In[7]:


raw_data.dtypes


# In[8]:


raw_data['EXAM1'].describe()


# In[9]:


raw_data['EXAM2'].describe()


# In[10]:


raw_data['EXAM3'].describe()


# In[12]:


raw_data['FINAL'].describe()


# In[13]:


raw_data['Unnamed: 4'].describe()


# In[15]:


raw_data['Unnamed: 5'].describe()


# In[16]:


raw_data['Unnamed: 6'].describe()


# In[17]:


raw_data['Unnamed: 7'].describe()


# In[18]:


raw_data['Unnamed: 8'].describe()


# In[31]:


def CalculateGradeRanges(Marks):
 if Marks<45 :
  return 'D'
 if Marks >= 45 and Marks < 55:
  return 'C'
 if Marks >= 55 and Marks < 75:
  return 'B'
 if Marks>=75 :
  return 'A'
  return 'Other'


# In[32]:


CalculateGradeRanges(67)


# In[33]:


grade_ranges = raw_data['FINAL'].apply(CalculateGradeRanges)


# In[34]:


grade_ranges 


# In[35]:


grade_ranges.value_counts()


# In[36]:


grade_ranges = raw_data['EXAM1'].apply(CalculateGradeRanges)


# In[37]:


grade_ranges.value_counts()


# In[38]:


grade_ranges = raw_data['EXAM2'].apply(CalculateGradeRanges)


# In[39]:


grade_ranges.value_counts()


# In[40]:


grade_ranges = raw_data['EXAM1'].apply(CalculateGradeRanges)


# In[41]:


grade_ranges.value_counts()


# In[42]:


f = plt.figure()
grade_ranges.value_counts().plot.pie(autopct='%1.0f%%',)
plt.title('Pie Chart of grade amount')


# In[44]:


pur = raw_data['FINAL'].value_counts()
pur.plot(kind='bar')


# In[45]:


pur = raw_data['EXAM1'].value_counts()
pur.plot(kind='bar')


# In[55]:


y = store_data.iloc[:,3]


# In[56]:


y


# In[63]:


x = raw_data [['EXAM1','EXAM2','EXAM3']]


# In[64]:


x.head()


# In[65]:


model = LinearRegression()
model.fit(x, y)


# In[66]:


x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()


# In[ ]:





# In[ ]:




