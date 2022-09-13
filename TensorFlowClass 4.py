#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as pt


# In[3]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[14]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale


# In[16]:


print("Feature matrix:", x_train.shape)
print("Target matrix:", x_test.shape)
print("Feature matrix:", y_train.shape)
print("Target matrix:", y_test.shape)


# In[17]:


fig, ax = pt.subplots(10, 10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train[k].reshape(28, 28),aspect='auto')
        k += 1
pt.show()


# In[19]:


model = Sequential([Flatten(input_shape=(28, 28)),
                    Dense(256, activation='sigmoid'),
                    Dense(128, activation='sigmoid'),
                    Dense(10, activation='sigmoid'),
                   ])


# In[30]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[33]:


model.fit(x_train, y_train, epochs=100, batch_size=2000,validation_split=0.2)


# In[37]:


results = model.evaluate(x_test, y_test, verbose = 0)
print('test loss, test acc:', results)


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




