#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv1D ,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[12]:


boston = load_boston()
x,y = boston.data,boston.target
print(x.shape)


# In[13]:


x= x.reshape(x.shape[0],x.shape[1],1)
print(x.shape)


# In[14]:


xtrain, xtest,ytrain, ytest=train_test_split(x,y,test_size=0.15)


# In[31]:


model=Sequential()
model.add(Conv1D(32,2,activation="relu", input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse",optimizer="adam")
model.summary()


# In[32]:


model.fit(xtrain, ytrain,batch_size=12,epochs=200,verbose=0)


# In[17]:


ypred=model.predict(xtest)


# In[19]:


print(model.evaluate(xtrain,ytrain))
print("MSE: %.4f" % mean_squared_error(ytest,ypred))


# In[28]:


x_ax = range(len(ypred))
plt.scatter(x_ax,ytest, s=5,color="blue", label="original")
plt.scatter(x_ax,ypred, lw=0.8,color="red", label="predicted")
plt.legend()
plt.show()


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




