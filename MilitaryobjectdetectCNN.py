#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import keras
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from keras.models import load_model
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
import cv2
from keras.utils import to_categorical


# In[2]:


def load_mstar_dataset(data_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                image = Image.open(image_path)
                image = np.array(image)
                images.append(image)
                labels.append(class_name)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# In[3]:


data_dir = r'C:\Users\rosha\Downloads\Padded_imgs'
X, y = load_mstar_dataset(data_dir)


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

print('Training images:')
print(X_train.shape)
print('Testing images:')
print(X_test.shape)


# In[5]:


# Display a subset of the training images
for i in range(0, 9):
    plt.subplot(330 + 1 + i)  # denotes 3x3 and position
    img = X_train[i + 50]  # no need to transpose else transpose([1,2,0])
    plt.imshow(img)

plt.show()


# In[6]:


print(X_train[0].shape)#should be and is 32x32x3


# In[7]:


for i in range(0,9):
    plt.subplot(330+1+i)#denotes 3x3 and postion
    img=X_train[i+50]#no need to transpose else transpose([1,2,0])
    plt.imshow(img)
    
plt.show()


# In[8]:


print(X_train[0])


# In[9]:


from sklearn.model_selection import train_test_split

seed = 42
np.random.seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float16')
X_test = X_test.astype('float16')
X_train = X_train / 255.0
X_test = X_test / 255.0


# In[10]:


print(X_train[0])


# In[22]:


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print(y_train.shape)
print(y_train[0])
print(y_train.min())
print(y_train.max())


# In[23]:


#encode outputs
Y_train=np_utils.to_categorical(y_train)
Y_test=np_utils.to_categorical(y_test)
num_classes=Y_test.shape[1]

print(Y_train.shape)
print(Y_train[0])


# In[24]:


from keras.models import Sequential
from keras.layers import Dropout,Activation,Conv2D,GlobalAveragePooling2D
#conv2d is the main convulational layer
from keras.optimizers import SGD#stochastic gradient descent 


# In[25]:


def allcnn(weights=None):
    #taking random weights ny default else usr passed pretrained weights
    
    model=Sequential()#we will be adding one layer after another
    
    #not the input layer but need to tell the conv. layer to accept input
    model.add(Conv2D(96,(3,3),padding='same',input_shape=(32,32,3)))#32x32x3 channels
    model.add(Activation('relu'))#required for each conv. layer
    model.add(Conv2D(96,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96,(3,3),padding='same',strides=(2,2)))
    model.add(Dropout(0.5))#drop neurons randomly;helps the network generalize(prevent overfitting on training data) better so instead of having individual neurons 
    #that are controlling specific classes/features, the features are spread out over the entire network
    
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding='same',strides=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(1,1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10,(1,1),padding='valid'))
    
    # add GlobalAveragePooling2D layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    #load the weights,if passed
    if weights:
        model.load_weights(weights)
    
    #return model
    return model


# In[26]:


#define the hyper parameters(generic or do Grid Search)
learning_rate=0.01
weight_decay=1e-6
momentum=0.9

#define training parameters
epochs=350 #from research paper
batch_size=32#run 32 images times then update the parameters instead of updating them after every image

model=allcnn()

#define optimizer and compile model
sgd=SGD(lr=learning_rate,decay=weight_decay,momentum=momentum,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

print(model.summary())#1.3m parameters and all are trainable

# #fit the model(update the parameters and loss)
# model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=epochs,batch_size


# In[27]:


get_ipython().system('pip install opencv-python')


# In[28]:


import cv2
from keras.utils import to_categorical

resized_test_images = []
for img in X_test:
    img = img.astype('uint8')  # Convert image to uint8 data type
    resized_img = cv2.resize(img, (32, 32))
    resized_test_images.append(resized_img)

# define hyperparameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# define weights and build model
weights = r'C:\Users\rosha\Downloads\weigh.hdf5'  # KERAS format hdf5
# pretrained weights that have already gone through the above process
model = allcnn(weights)

# define optimizer and compile model
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# print model summary
print(model.summary())

# Resize test images to match the input shape of the model
resized_test_images = []
for img in X_test:
    img = img.astype('uint8')  # Convert image to uint8 data type
    resized_img = cv2.resize(img, (32, 32))
    resized_test_images.append(resized_img)

# Convert the resized images to numpy array
X_test_resized = np.array(resized_test_images)

# Normalize the resized test images
X_test_resized = X_test_resized.astype('float32')
X_test_resized = X_test_resized / 255.0

# Perform one-hot encoding on the ground truth labels
num_classes = 10  # Replace with the correct number of classes
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

# Test the model with the resized test images
scores = model.evaluate(X_test_resized, y_test_encoded, verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100))


# In[21]:


# Resize the images in the batch to the desired input shape
resized_images = []
for img in batch:
    # Convert the image to the appropriate data type (e.g., uint8)
    img = np.array(img, dtype=np.uint8)
    resized_img = cv2.resize(img, (32, 32))
    resized_images.append(resized_img)

resized_batch = np.array(resized_images)

# Make predictions on the resized batch
predictions = model.predict(resized_batch, verbose=1)

# Assuming you have defined the class_labels dictionary
classes = range(0, 10)  # 10 not included
names = ['ZSU_23_4', 'ZIL131', 'T62', 'SLICY', 'D7', 'BTR_60', 'BRDM_2', '2S1', 'ship']
class_labels = dict(zip(classes, names))
print(class_labels)

# Print the predicted class labels
predicted_labels = np.argmax(predictions, axis=1)
for label in predicted_labels:
    print(class_labels[label])


# In[17]:


#these are individual class probabilities, should sum to 1.0
for image in predictions:
    print(np.sum(image))

#shows that there is hundred percent probability that images to belong to one of the c


# In[18]:


# use np.argmax() to convert class probabilities to class labels
class_result=np.argmax(predictions,axis=-1)
print(class_result)


# In[19]:


# Create a grid of 3x3 images
fig, axs = plt.subplots(3, 3, figsize=(15, 6))
fig.subplots_adjust(hspace=1)
axs = axs.flatten()

for i, img in enumerate(batch):
    # Convert the image to uint8
    img = (img * 255).astype(np.uint8)
    
    # Determine label for each prediction and set the title
    for key, value in class_labels.items():
        if class_result[i] == key:
            title = 'Prediction: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)
    
    # Plot the image
    axs[i].imshow(img)
    
# Show the plot
plt.show()     


# In[ ]:




