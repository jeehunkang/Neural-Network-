#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Simple 


# In[ ]:


#importing necessary libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras


data = keras.datasets.fashion_mnist 


(train_images, train_labels), (test_images, test_labels) = data.load_data()

# does this automatically set the ratio for training and testing? 


class_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'sandal','shirt','sneaker','bag', 'angle boot']


# sizing down the data 
train_images = train_images/255.0
test_images = test_images/255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation ="relu"),
    # 128 means, it's going to spit out 128 dimensional output 
    keras.layers.Dense(10, activation = 'softmax'),
    # 10 means, it's going to spit out 10 dimensional output 

    
])


# applying optimizer to make non-linearity & choose loss function.
model.compile(optimizer = "adam", loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 5)

# now, check the accuracy on test dataset 
#test_loss, test_acc = model.evaluate(test_images, test_labels)


prediction = model.predict(test_images)

# amongst the 10 last hidden layers, argmax will predict the one with the highest probability,
# which will be the type of clothes.
print(class_names[np.argmax(prediction[0])])


# print out 5 images showing images along with their predicted outcome and actual outcome 
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("Actual: ", class_names[test_labels[i]])
    plt.title("Prediction", class_name[np.argmax(prediction[i])])
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




