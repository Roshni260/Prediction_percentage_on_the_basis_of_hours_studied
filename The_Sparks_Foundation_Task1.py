#!/usr/bin/env python
# coding: utf-8

# ## Hello Friends!! My name is Roshni Sanjay Jha

# ## The Sparks Foundation

# ### Task1: Prediction Using Supervised ML 
# 
# ### Predicting the percentage of an student based on the no. of study hours.

# <b>Importing Libraries</b>

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <b>Exploring the dataset</b>

# In[3]:


data= pd.read_csv("http://bit.ly/w-data")
data.shape


# In[4]:


data.head()


# <b>Plotting the distribution of scores using 2-D graph</b>

# In[5]:


data.plot(x='Hours', y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage')
plt.show()


# <b>Preparing the data</b>

# In[6]:


X= data.iloc[:, :-1].values  #attributes
y= data.iloc[:, 1].values   #label


# <b> Splitting the dataset into training and testing </b>

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)


# <b>Training the Algorithm</b>

# In[12]:


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train) 

print('Model is trained!😊')


# <b> Plotting the regression line</b>

# In[13]:


line = regressor.coef_*X+ regressor.intercept_

plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# <b> Making Predictions</b>

# In[14]:


print(X_test)
y_pred= regressor.predict(X_test)


# In[15]:


df = pd.DataFrame({'Actual': y_test,'Predicted':y_pred})
df


# <b> Predicting percentage on the basis of no.of hours studied</b>

# In[26]:


hours = 9.25
hr = [[9.25]]
own_pred = regressor.predict(hr)
print("No of Hours studied={}".format(hours))
print("Predicted percentage={}".format(own_pred[0]))


# In[28]:


hours = 5
hr=[[5]]
own_pred = regressor.predict(hr)
print("No of Hours studied={}".format(hours))
print("Predicted percentage={}".format(own_pred[0]))


# <b> Evaluating the Model</b>

# In[29]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))

