# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
try:
#   %tensorflow_version 2.x # enable TF 2.x in Colab
except Exception:
  pass

import tensorflow as tf
print(tf.__version__)

# Commented out IPython magic to ensure Python compatibility.
#Ccheck if using GPU
# %tensorflow_version 2.x
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#predict
# %% Import packages
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.optimizers import Adam
import h5py



# %% Import
# Import data
#x = np.load('sim_ecg_data_new.npy')
# Import model
base_model = load_model("/content/gdrive/MyDrive/Colab Notebooks/model.hdf5",compile = False)
base_model._name = 'nnnn'
#model.compile(loss='binary_crossentropy', optimizer=Adam())
#y_score = model.predict(x, batch_size=32, verbose=1)

# Generate dataframe
#np.save("dnn_output_paper.npy", y_score)

###1. Model Building
# Freeze the first 5 layers

for i in range(5):
    base_model.layers[i].trainable = False

for i in range(5,50):
    base_model.layers[i].trainable = True

base_model.summary()

inputs = base_model.inputs

## Add 3 conv1d layers and three dense layers after the model structure

from keras.models import Model
from keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                          BatchNormalization, Activation, Add,
                          Flatten, Dense)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
ll = base_model.layers[45].output
ll = tf.keras.layers.Conv1D(64, 2, activation='relu', input_shape=(16, 320))(ll)
#ll = MaxPooling1D(pool_size=2)(ll)
ll = tf.keras.layers.Conv1D(64, 2, activation='relu')(ll)
ll = tf.keras.layers.Conv1D(64, 2, activation='relu')(ll)
#ll = MaxPooling1D(pool_size=2)(ll)
#ll = tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(16, 320))(ll)
#ll = MaxPooling1D(pool_size=2)(ll)
ll = Flatten()(ll)
ll = Dense(32,activation='relu')(ll)
ll = Dense(64,activation='relu')(ll)
ll = Dense(128,activation='relu')(ll)
diagn = Dense(1,activation='sigmoid')(ll)
'''
ll = Flatten()(ll)
ll = Dense(256,activation='sigmoid')(ll)
ll = Dense(128,activation='sigmoid')(ll)
ll = Dense(64,activation='sigmoid')(ll)
ll = Dense(32,activation='sigmoid')(ll)
diagn = Dense(1,activation='sigmoid')(ll)
'''
trans_model = Model(inputs,diagn)

trans_model.summary()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy')
lr = 0.0001 # lower

batch_size = 64
opt = Adam(lr)

trans_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(),
                           tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

### 2. Loading Data
X_train= np.load('/data/Xtrain.npy')
y_train=np.load('/data/ytrain.npy')
X_test=np.load('/data/Xtest.npy')
y_test=np.load('/data/ytest.npy')

from autoecg_model_gpu import preprocess
X_train=preprocess(X_train)
X_test=preprocess(X_test)
print(X_train.shape,X_test.shape)

### 3. Fitting model
import keras
from datetime import datetime
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

trans_model.load_weights('transfermodel.h5')

history_2 = trans_model.fit(X_train,
                          y_train,
                          batch_size=64,
                          epochs=50,
                          validation_data=(X_test, y_test),
                          callbacks=[tensorboard_callback]
                         )

### 4. Result plotting
import matplotlib.pyplot as plt
def plot(history):
  
  # The history object contains results on the training and test
  # sets for each epoch
  acc = history.history['binary_accuracy']
  val_acc = history.history['val_binary_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  # Get the number of epochs
  epochs = range(len(acc))

  plt.title('Training and validation accuracy')
  plt.plot(epochs, acc, color='blue', label='Train')
  plt.plot(epochs, val_acc, color='orange', label='Val')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  _ = plt.figure()
  plt.title('Training and validation loss')
  plt.plot(epochs, loss, color='blue', label='Train')
  plt.plot(epochs, val_loss, color='orange', label='Val')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

plot(history_2)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs
