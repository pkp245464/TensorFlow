#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pylab as plt


# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()


# In[3]:


train_images.shape


# In[4]:


len(train_labels)


# In[5]:


test_images.shape


# In[36]:


len(test_labels)


# In[25]:


train_labels


# In[6]:


class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


# In[7]:


plt.matshow(test_images[0])


# In[8]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True)
plt.show()


# In[9]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# In[10]:


plt.figure(figsize=(20,20))
for i in range(25):
    plt.subplot(5,5,i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[11]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
])


# In[18]:


model.compile(optimizer='sgd',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


# In[19]:


model.fit(train_images,train_labels,epochs=10)


# In[21]:


test_loss,test_acc = model.evaluate(test_images,test_labels,verbose = 2)
print("\nTest Accuracy: ",test_acc)


# In[30]:


probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


# In[31]:


predictions[0]


# In[32]:


test_labels[0]


# In[33]:


def plot_image(i, predictions_array, true_label,img):
  true_label, img= true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label= np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color='red'
    plt.xlabel ("{} {:2.0f} % ({})". format(class_names[predicted_label],
                                              100*np.max(predictions_array),
                                              class_names[true_label]),color=color)
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot= plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0,1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[34]:


i=0
plt.figure (figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],test_labels)
plt.show()
i=12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()


# In[35]:


img = test_images[1]
print(img.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


initia_model = keras.Sequential(
[
    keras.Input(shape=(250,250,3)),
    layers.Conv2D(32,5,strides=2,activation="relu"),
    layers.Conv2D(32,3,activation="relu"),
    layers.Conv2D(32,3,activation="relu"),
])
feature_extractor = keras.Model(
    inputs = initia_model.inputs,
    outputs = [layer.output for layer in initia_model.layers],
)
x = tf.ones((1,250,250,3))
features = feature_extractor(x)
features


# In[16]:


initia_model = keras.Sequential(
[
    keras.Input(shape=(250,250,3)),
    layers.Conv2D(32,5,strides=2,activation="relu"),
    layers.Conv2D(32,3,activation="relu",name = "my_intermediate_layer"),
    layers.Conv2D(32,3,activation="relu"),
])
feature_extractor = keras.Model(
    inputs = initia_model.inputs,
    outputs = initia_model.get_layer(name="my_intermediate_layer").output,
)
x = tf.ones((1,250,250,3))
features = feature_extractor(x)
features


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




