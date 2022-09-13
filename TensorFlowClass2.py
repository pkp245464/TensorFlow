#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Ragged Tensor and String Tensor
import tensorflow as tf
digits = tf.ragged.constant([[3,1,4,1],[],[5,9,2],[6],[]])
words = tf.ragged.constant([["so","long"],["thanks","for","all","the","fish"]])

print(tf.add(digits,3))
print(tf.reduce_mean(digits,axis=1))
print(tf.concat([digits,[[5,3]]],axis=0))
print(tf.tile(digits,[1,2]))
print(tf.strings.substr(words,0,2))

print(tf.map_fn(tf.math.square,digits))


# In[6]:


#String Split
s = tf.constant("PANKAJ KUMAR") 
print(tf.strings.split(s,' '))


# In[9]:


#Type Casting
a_string = tf.strings.bytes_split(tf.constant("Duck"))
b_int = tf.io.decode_raw(tf.constant("Duck"),tf.uint8)
print("a String",a_string)
print("b int",b_int)


# In[15]:


#Sparse tensor
sparse_tensor = tf.sparse.SparseTensor(indices=[[0,0],[1,2]],values=[1,2],dense_shape=[3,4])
#print(sparse_tensor)

print(tf.sparse.to_dense(sparse_tensor))


# In[19]:


c = tf.constant([[4.0,5.0],[10.0,1.0]])
#find the largest number
print(tf.reduce_max(c))
#find the index of largest number index
print(tf.argmax(c))
#compute the softmax
print(tf.nn.softmax(c))


# In[24]:


#indexing
rank_1_tensor = tf.constant([0,1,1,2,3,5,8,13,21,34])
#print(rank_1_tensor)

print("First: ",rank_1_tensor[0].numpy())
print("Second: ",rank_1_tensor[1].numpy())
print("Third: ",rank_1_tensor[2].numpy())


# In[27]:


#Single Indexing index
print("EveryThings: ",rank_1_tensor[:].numpy())
print("Before 4: ",rank_1_tensor[:4].numpy())
print("from 4 to the end",rank_1_tensor[4:].numpy())
print("from 2,before 7: ",rank_1_tensor[2:7].numpy())
print("Every other item",rank_1_tensor[::2].numpy())
print("Reversed",rank_1_tensor[::-1].numpy())


# In[28]:


rank_2_tensor = tf.constant([[1,2],[3,4],[5,6]],dtype=tf.float16)
print(rank_2_tensor)


# In[30]:


print("Second row: ",rank_2_tensor[1,:].numpy())
print("Second Column: ",rank_2_tensor[:,1].numpy())
print("Last row",rank_2_tensor[-1,:].numpy())
print("Last Column: ",rank_2_tensor[:,-1].numpy())
print("First item in last column: ",rank_2_tensor[0,-1].numpy())
print("Skip the first row: ",rank_2_tensor[1:,:].numpy(),"\n")


# In[31]:


#Manipulation Shape
var_x = tf.Variable(tf.constant([[0],[1],[2]]))
print(var_x.shape)


# In[32]:


print(var_x.shape.as_list())


# In[33]:


reshaped = tf.reshape(var_x,[1,3])
print(var_x.shape)
print(reshaped.shape)


# In[ ]:


#flatten 


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




