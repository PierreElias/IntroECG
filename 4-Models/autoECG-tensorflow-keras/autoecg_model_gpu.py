# -*- coding: utf-8 -*-


# Turn off GPU if needed
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

##Checking if using GPU
import tensorflow as tf
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
tf.config.experimental.list_physical_devices('GPU')

"""# SET UP 

"""

!pip install tensorflow-gpu==2.3.0

!pip install Keras==2.4.3

import keras 
import tensorflow as tf
print(keras.__version__,tf.__version__)

###############################
####### Start from here #######
###############################

import keras
import tensorflow as tf
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.optimizers import Adam
import h5py

import IPython
!pip install -q -U keras-tuner
import kerastuner as kt

# Commented out IPython magic to ensure Python compatibility.

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

"""# Load Data & Preprocessing

"""

X_train= np.load('/data/Xtrain.npy')
y_train=np.load('/data/ytrain.npy')

X_test=np.load('/data/Xtest.npy')
y_test=np.load('/data/ytest.npy')

#X_train=X_train[:,0,:,:]
#X_test=X_test[:,0,:,:]

import numpy as np
def preprocess(X):
  m=4096-X.shape[2]
  y=np.pad(X_train,[(0,0),(0,m),(0,0)],mode='constant', constant_values=0)
  return y

X_train=preprocess(X_train)
X_test=preprocess(X_test)
print(X_train.shape,X_test.shape)

"""## Residual Net Model 

"""

from keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                          BatchNormalization, Activation, Add,
                          Flatten, Dense)
from keras.models import Model
import numpy as np


class ResidualUnit(object):
    """Residual unit block (unidimensional).
    Parameters
    ----------
    n_samples_out: int
        Number of output samples.
    n_filters_out: int
        Number of output filters.
    kernel_initializer: str, optional
        Initializer for the weights matrices. See Keras initializers. By default it uses
        'he_normal'.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. Default is 17.
    preactivation: bool, optional
        When preactivation is true use full preactivation architecture proposed
        in [1]. Otherwise, use architecture proposed in the original ResNet
        paper [2]. By default it is true.
    postactivation_bn: bool, optional
        Defines if you use batch normalization before or after the activation layer (there
        seems to be some advantages in some cases:
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
        If true, the batch normalization is used before the activation
        function, otherwise the activation comes first, as it is usually done.
        By default it is false.
    activation_function: string, optional
        Keras activation function to be used. By default 'relu'.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027 [cs], Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
                 dropout_rate=0.5, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x
   
    def on_epoch_end(self, epoch, logs=None):
        print('###########',Keras.eval(self.model.optimizer.lr))

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        #n_samples_in = y.shape[1].value
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        #n_filters_in = y.shape[2].value
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = Conv1D(self.n_filters_out, self.kernel_size, padding='same',
                   use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
                   padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


# ----- Model ----- #
kernel_size = 8
kernel_initializer = 'he_normal'
signal = Input(shape=(4096, 12), dtype=np.float32, name='signal')
age_range = Input(shape=(6,), dtype=np.float32, name='age_range')
is_male = Input(shape=(1,), dtype=np.float32, name='is_male')
x = signal
x = Conv1D(64, kernel_size, padding='same', use_bias=False,
           kernel_initializer=kernel_initializer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, x])
x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, y])
x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, y])
x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, y])
x = Flatten()(x)
#diagn = Dense(6, activation='sigmoid')(x)
diagn = Dense(1,activation='sigmoid')(x)
model = Model(signal, diagn)

model.save_weights('model.h5')

model.summary()

import tensorflow as tf
bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy')
lr =  0.01
opt = tf.keras.optimizers.Adam(lr,epsilon=0.1)
loss = bce_logits
lr_metric = opt.lr
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc', lr_metric])

import keras
from datetime import datetime
#from sklearn.metrics import confusion_matrix, f1_score, 
def scheduler(epoch, lr):
    if epoch <= 7:
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr
    else:
        tf.summary.scalar('learning rate', data=lr * 0.9, step=epoch)
        return lr * 0.9

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy')
lr =  0.005
opt = tf.keras.optimizers.Adam(lr,epsilon=0.1)
loss = bce_logits
lr_metric = opt.lr

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(),
                           tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model.fit(X_train,
          y_train,
          batch_size=64,
          epochs=50,
          validation_data=(X_test, y_test),
          callbacks=[callback,tensorboard_callback])

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

"""# Hyperparameter Tuning

"""

from kerastuner import HyperModel
from keras import Sequential
from keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                          BatchNormalization, Activation, Add,
                          Flatten, Dense)
from keras.models import Model
import numpy as np

class ECGhyper(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape



    def build(self, hp):
        '''
        kernel_initializer = 'he_normal'
        model = Sequential()
        model.add(
            Conv1D(64,
                kernel_size=hp.Choice('kernel_size', [8,16,24,38], default=8),
                input_shape=input_shape
            )
        )
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(
            ResidualUnit(1024, 128,
                kernel_size=hp.Choice('kernel_size', [8,16,24,38], default=8),
                kernel_initializer=kernel_initializer
            )
        )
        
        model.add(
            ResidualUnit(256, 196, kernel_size=hp.Choice('kernel_size', [8,16,24,38], default=8),
                    kernel_initializer=kernel_initializer))
        
        model.add(
            ResidualUnit(64, 256, kernel_size=hp.Choice('kernel_size', [8,16,24,38], default=8),
                    kernel_initializer=kernel_initializer))
          
        model.add(
            ResidualUnit(16, 320, kernel_size=hp.Choice('kernel_size', [8,16,24,38], default=8),
                    kernel_initializer=kernel_initializer))
        
        model.add(Flatten())

        model.add(Dense(1,activation='sigmoid'))
        '''
        #kernel_size = 16
        kernel_size=hp.Choice('kernel_size', [8,16,24,38], default=8)
        kernel_initializer = 'he_normal'
        signal = Input(shape=(4096, 12), dtype=np.float32, name='signal')
        x = signal
        x = Conv1D(64,
                kernel_size=kernel_size,
                input_shape=input_shape,kernel_initializer=kernel_initializer
            )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x, y = ResidualUnit(1024, 128,
                kernel_size=kernel_size,
                kernel_initializer=kernel_initializer
            )([x, x])
        x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, y])
        x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, y])
        x, _ = ResidualUnit(kernel_size, 320, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, y])
        x = Flatten()(x)
        #diagn = Dense(6, activation='sigmoid')(x)
        diagn = Dense(1,activation='sigmoid')(x)
        model = Model(signal, diagn)

        lr = hp.Choice('lr', values = [1e-2, 1e-3, 1e-4]) 
        


        


        #bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy')
        opt = tf.keras.optimizers.Adam(lr,epsilon=0.1)
        #loss = bce_logits
        #lr_metric = opt.lr

        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(), 'AUC',
                                  'Precision', 'Recall'])

        
        return model

input_shape=(4096, 12)
hypermodel=ECGhyper(input_shape)

from kerastuner import RandomSearch, Objective
tuner_rs = RandomSearch(
            hypermodel,
            objective=Objective("val_loss", direction="max"),
            seed=42,
            max_trials=10,
            executions_per_trial=2)

import keras
from datetime import datetime
#from sklearn.metrics import confusion_matrix, f1_score, 
def scheduler(epoch, lr):
    if epoch <= 7:
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr
    else:
        tf.summary.scalar('learning rate', data=lr * 0.9, step=epoch)
        return lr * 0.9

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

tuner_rs.search(X_train, y_train, epochs=10, validation_split=0.2,callbacks=[callback,tensorboard_callback])

best_model = tuner_rs.get_best_models(num_models=1)[0]
#loss, mse = best_model.evaluate(x_test_scaled, y_test)

best_model.fit(X_train,
          y_train,
          batch_size=64,
          epochs=30,
          validation_data=(X_test, y_test),
          callbacks=[callback,tensorboard_callback])

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs

model.summary()

model.save_weights('model.h5')

#####RESET MODEL######
model.load_weights('model.h5')

