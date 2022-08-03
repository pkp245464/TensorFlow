#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf


# In[87]:


#Constant In TensorFlow


# In[44]:


a = tf.constant(1)
a


# In[45]:


b = tf.constant(True)
b


# In[46]:


c = tf.constant("PANKAJ")
c


# In[47]:


d = tf.constant(3.81)
d


# In[ ]:


#creating numpy array


# In[30]:


import numpy as np


# In[49]:


np_array1 = tf.constant([1,2,3,4,5])
np_array1


# In[37]:


np_array2 = tf.constant([[1,2,3,4,5],[1,2,3,4,5]])
np_array2


# In[43]:


np_array3 = tf.constant([[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]])
np_array3


# In[50]:


np_array4 = tf.constant([1.1,2.2,3.3,4.4,5.5])
np_array4


# In[53]:


np_array5 = tf.constant(np.array([1,2,3.3,4.4,"PANKAJ"]))
np_array5
#all the entities converted into the string because of higher priorities


# In[54]:


np_array5 = tf.constant(np.array([1,2,3.3,4.4,3,7,9.27]))
np_array5
#all the entities converted into the float because of higher priorities


# In[ ]:


#how to create 2D array constant tensor


# In[55]:


np_one = tf.constant([[1,2],[3,4]])
np_one


# In[83]:


#convert array[1,2,3,4,5,6,7,8,9] into 3x3 matrix
np_array6 = tf.constant([1,2,3,4,5,6,7,8,9],shape = (3,3))
np_array6


# In[84]:


type(np_array6)


# In[86]:


np_array6.shape


# In[88]:


#Variable In TensorFlow


# In[92]:


v1 = tf.Variable(1)
v1


# In[93]:


v2 = tf.Variable("PANKAJ")
v2


# In[94]:


v2 = tf.Variable(3.9)
v2


# In[95]:


v3 = tf.Variable(True)
v3


# In[96]:


v4 = tf.Variable([1,2,3,4,5])
v4


# In[97]:


v4.name


# In[98]:


v4.shape


# In[103]:


v5 = tf.Variable([[1,2],[3,4]],shape = (2,2),dtype = "float32")
v5


# In[109]:


#create q complex Variable Tensor 
# i.c  i + 2j

complexVar = tf.Variable([2 + 3j])
complexVar


# In[110]:


#create TF from constant TensorFlow Variable

t_conn = tf.Variable([1,2,3,4,5])
t_conn


# In[111]:


tf.Variable(t_conn)


# In[112]:


# viewed / convert as tensor


# In[113]:


a


# In[114]:


tf.convert_to_tensor(a)


# In[115]:


#change / assign a new value to tensor


# In[126]:


t_conn


# In[127]:


t_conn.assign([6,7,8,9,10])


# In[ ]:





# In[ ]:




