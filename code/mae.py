#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[6]:


y_predicted = np.random.rand(5)
y_true = np.full(5,1,dtype=float)

print(y_predicted)
print(y_true)


# In[17]:


#mean absolute error and mean squred error
def mae(y_predicted,y_true):
    total_error = 0
    for i in range(0,len(y_true)):
        total_error += abs(y_predicted[i]-y_true[i])
    return total_error/len(y_true)

y = mae(y_predicted,y_true)
print(y)
print()

#now by using the numpy function
y = np.mean(abs(y_predicted-y_true))
print(y)


# In[ ]:





# In[22]:


#now doing the mean squred error
def mse(y_predicted,y_true):
    total_error = 0
    for i in range(0,len(y_true)):
        total_error += (y_predicted[i]-y_true[i])**2
    return total_error/len(y_true)
y = mse(y_predicted,y_true)
print(y)
print()
#now using the numpy array
y = np.mean(np.square(y_predicted-y_true))
print(y)


# In[24]:


def loss_function(y_predicted,y_true):
    epsilon = 1e-15
    y_predicted_new = np.where(y_predicted==0,epsilon,y_predicted)
    y_predicted_new = np.where(y_predicted==1,1-epsilon,y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))

y = loss_function(y_predicted,y_true)
print(y)

