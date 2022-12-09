#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import tensorflow_datasets as tfds


# In[4]:


dataset,info = tfds.load('imdb_reviews/subwords8k',with_info=True,as_supervised=True)


# In[6]:


train_dataset,test_dataset = dataset['train'],dataset['test']


# In[10]:


encoder = info.features['text'].encoder


# In[11]:


print(encoder.vocab_size)


# In[12]:


sample_strings = "Hello Tensorflow"


# In[14]:


encoded_string = encoder.encode(sample_strings)
print(encoded_string)


# In[17]:


decoded_string = encoder.decode(encoded_string)
print(decoded_string)


# In[18]:


buffer_size = 10000
batch_size = 64


# In[19]:


train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.padded_batch(batch_size)
test_dataset = test_dataset.padded_batch(batch_size)


# In[20]:


from tensorflow import keras
from keras import layers


# In[21]:


model = keras.Sequential([
                          layers.Embedding(encoder.vocab_size,64),
                          layers.Bidirectional(layers.LSTM(64)),
                          layers.Dense(64,activation='relu'),
                          layers.Dense(1)
])


# In[22]:


model.compile(optimizer='adam',loss=keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])


# In[ ]:


model.fit(train_dataset,epochs=20,validation_data=test_dataset)


# In[ ]:


test_loss,test_acc = model.evaluate(test_dataset)


# In[ ]:


sample_test = ('The movie was cool. The animation and the graphics were out of this world. I would recommend this movie')


# In[ ]:


encoded_test = encoder.encode(sample_test)


# In[ ]:


enoded_test


# In[ ]:


prediction = model.predict(tf.expand_dims(encoded_test,0))


# In[ ]:


if prediction >= 0.5:
    print("Postive")
else:
    print("Negative")


# In[ ]:


model = keras.Sequential([
                          layers.Embedding(encoder.vocab_size),
                          layers.Bidirectional(layers.LSTM(,return_sequences = True)),
                          layers.Bidirectional(layers.LSTM(32)),
                          layers.Dense(64,activation='relu'),
                          layers.Dropout(0.5),
                          layers.Dense(1)
])


# In[ ]:





# In[ ]:




