import tensorflow.keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from PIL import Image
#from numpy import asarray
import numpy as np
from tensorflow.keras import backend as K
   
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

from tensorflow.keras import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras import models
import tensorflow as tf
from scipy.ndimage import interpolation

loss_tracker = tensorflow.keras.metrics.Mean(name="loss")
mae_metric = tensorflow.keras.metrics.MeanAbsoluteError(name="mae")

def getAssembledModel(patchSize):
    input_img = tf.keras.Input(shape=(patchSize, patchSize, 3))

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = ProposedModel(input_img, decoded)
    autoencoder.compile(optimizer="adam")

    autoencoder.summary()
    
    return autoencoder

class ProposedModel(tf.keras.Model):
    model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
    #model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
    model.summary()        
    def train_step(self, data):
        x, y = data
        a1 = 0.7
        a2 = 0.3
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            #print("########################## Coming Here ##########################")
            # Compute stylization loss
            yy = self.model(x)
            yp = self.model(y_pred)
            loss1 = tf.math.reduce_mean(a1 * tf.keras.losses.mean_absolute_error(yy, yp),(1,2))
            loss2 = tf.math.reduce_mean(a2 * tf.keras.losses.mean_squared_error(y, y_pred),(1,2))
            #print(loss1)
            #print(loss2)
            loss = loss1 + loss2

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

    def test_step(self, data):
        x, y = data
        yPred = self(x, training=False)
        loss = tf.keras.losses.mean_squared_error(y, yPred)
        
        # Compute our own metrics
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, yPred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, mae_metric]