#!/usr/bin/env python
# coding: utf-8

# In[46]:


import tensorflow.keras as tf
import matplotlib.pyplot as plt
import numpy as np


# In[47]:


mnist = tf.datasets.mnist


# In[48]:


(xtrain,ytrain),(xtest,ytest) = mnist.load_data()


# In[49]:


xtrain.shape


# In[50]:


ytrain.shape


# In[51]:


plt.imshow(xtrain[59999],cmap='gray')


# In[52]:


plt.matshow(xtrain[59999])


# In[53]:


model = tf.models.Sequential()
model.add(tf.layers.Flatten())
model.add(tf.layers.Dense(784,activation="relu"))
model.add(tf.layers.Dense(10,activation="softmax"))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=["accuracy"])


# In[54]:


xtrain = xtrain/255
xtest = xtest/255


# In[56]:


model.fit(xtrain,ytrain,epochs=10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




