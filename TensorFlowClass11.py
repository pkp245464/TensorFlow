#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import os
import time
from tensorflow import keras
from keras import layers


# In[4]:


path_to_file = keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


# In[6]:


text = open(path_to_file,'rb').read().decode(encoding='utf-8')


# In[7]:


print(len(text))


# In[10]:


print(text[:250])


# In[11]:


vocab = sorted(set(text))


# In[12]:


print(len(vocab))


# In[13]:


char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


# In[19]:


text_as_int = np.array([char2idx[c] for c in text])


# In[20]:


print(repr(text[:10]),text_as_int[:10])


# In[21]:


seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


# In[22]:


for i in char_dataset.take(5):
    print(i.numpy(),idx2char[i.numpy()])


# In[23]:


sequences = char_dataset.batch(seq_length + 1,drop_remainder=True)


# In[25]:


for item in sequences.take(5):
    print(item.numpy())
    print(repr(''.join(idx2char[item.numpy()])))


# In[26]:


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text,target_text


# In[27]:


dataset = sequences.map(split_input_target)


# In[28]:


for input_example,target_example in dataset.take(1):
    print('input data: ',repr(''.join(idx2char[input_example.numpy()])))
    print('Target data: ',repr(''.join(idx2char[target_example.numpy()])))


# In[29]:


for i ,(input_idx,target_idx) in enumerate(zip(input_example[:5],target_example[:5])):
    print(i)
    print(input_idx,repr(idx2char[input_idx]))
    print(target_idx,repr(idx2char[target_idx]))


# In[ ]:


batch_size = 64


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




