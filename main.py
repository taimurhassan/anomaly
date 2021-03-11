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
from config import *
from model import ProposedModel, getAssembledModel

tf.config.run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)        
        
def getHeatMap(image, encoded):
    image_size = 224

    # Load pre-trained Keras model and the image to classify
    model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
    img_tensor = preprocessing.image.img_to_array(image)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)

    conv_layer = model.get_layer("block5_conv3")
    heatmap_model = encoded
    
    with tf.GradientTape() as gtape:
        conv_output = heatmap_model(tf.convert_to_tensor(img_tensor, dtype=tf.float32))
        print(conv_output)
        loss = predictions[:, np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    print(heatmap.shape)  

    return heatmap
	  
def getPatches(folder, isTraining, p):
    patches = []
    i_i = []
    i_j = []

    mean = 0
    var = 10
    sigma = var ** 0.5
    act_size = 2240
    gaussian = np.random.normal(mean, sigma, (act_size, act_size)) 
        
    doChunking = False

    index = 0
    i2 = 1
    for filename in os.listdir(folder):
        if isTraining == True:
            print(str(i2) + ", chunking training image '" + filename + "'")
        else:
            print(str(i2) + ", chunking test image '" + filename + "'")
        
        i2 = i2 + 1
        image = Image.open(folder + filename)
        data = np.array(image)
        
        if isTraining == True:
            # adding Gaussian noise            
            if len(data.shape) == 2:
                data = data + gaussian
            else:
                data[:, :, 0] = data[:, :, 0] + gaussian
                data[:, :, 1] = data[:, :, 1] + gaussian
                data[:, :, 2] = data[:, :, 2] + gaussian

        data = data.astype('float32') / 255.
        row, col,ch = data.shape
        
        for i in range(row):
            for j in range(col):
                if (i+1)*p <= row and (j+1)*p <= col:
                    patch = data[(i)*p:(i+1)*p,(j)*p:(j+1)*p,:]
                    patches.append(patch)
                    i_i.append(i)
                    i_j.append(j)
          
        if doChunking == True:
            if index >= 10:
                break
            else:
                index = index + 1
     
    patches = np.array(patches)
    
    return i_i, i_j, patches
    

def transferWeights(model1, model2):    
    for i in range(1,len(model1.layers)):
        model2.layers[i].set_weights(model1.layers[i].get_weights())
    
    return model2
    
autoencoder = getAssembledModel(p)
    
# ONE-TIME TRAINING
if doTraining == True:
    _, _,x_train = getPatches(tr_folder, True, p)
    _, _,x_valid = getPatches(te_folders[0], False, p) # using test dataset-1 for validation as well, it is user configurable
    
    print(x_train.shape)
    print(x_valid.shape)
    
    autoencoder.fit(x_train, x_train,
                    epochs=200,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(x_valid, x_valid))
    autoencoder.save("model.tf")
else:
    model1 = tf.keras.models.load_model("model.tf", compile=False)
    autoencoder = transferWeights(model1, autoencoder)
    
# TESTING
for d in range(numDatasets):
    i_i, i_j, x_test = getPatches(te_folders[d], False, p)

    print(x_test.shape)
    
    print("**********************Reconstructing Patches*******************")
    decoded_imgs = []   

    l1, r1,c1,ch1 = x_test.shape

    for i in range(l1):
        decoded_imgs.append(autoencoder.predict(x_test[i].reshape(1, p, p, 3)))

    decoded_imgs = np.array(decoded_imgs)               
                
    t, r, c, ch = x_test.shape

    d_imgs = []
    d_test = []
    heatmaps = []
    i = 0
    j = 0

    img = np.zeros((act_size, act_size, 3), dtype='float32')
    img2 = np.zeros((act_size, act_size, 3), dtype='float32')
    img3 = np.zeros((act_size, act_size, 3), dtype='float32')

    print(decoded_imgs.shape)

    row, col,ch = img.shape
    
    print("**********************Stitching Images*******************")
    for k in range(len(i_i)):
        patch = decoded_imgs[k].reshape(p, p, 3);
        i = i_i[k]
        j = i_j[k]
        img[(i)*p:(i+1)*p,(j)*p:(j+1)*p,:] = patch
        
        
    #    heatmap = getHeatMap(patch, keras.Model(input_img, encoded))
        img3[i*p:(i+1)*p,j*p:(j+1)*p,:] = x_test[k].reshape(p, p, 3)-patch 
        
        patch = x_test[k].reshape(p, p, 3);
        
        img2[i*p:(i+1)*p,j*p:(j+1)*p,:] = patch
        
        if i == 9 and j == 9:
            d_imgs.append(img)
            img = np.zeros((act_size, act_size, 3), dtype='float32')
            d_test.append(img2)
            img2 = np.zeros((act_size, act_size, 3), dtype='float32')    
            heatmaps.append(img3)
            img3 = np.zeros((act_size, act_size, 3), dtype='float32')        
    
    d_test = np.array(d_test)
    d_imgs = np.array(d_imgs)
    heatmaps = np.array(heatmaps)

    print(d_imgs.shape)
    print(d_test.shape)

    t, r, c, ch = d_imgs.shape 
    
    m = d
    folder = res_fake[m]
    print("**********************Saving reconstructed images at " + folder + "*******************")
    for i in range(t):
        A = (255 * d_imgs[i].reshape(act_size, act_size, 3)).astype(np.uint8)
        im = Image.fromarray(A)
        newsize = (224, 224) 
        #im = im.resize(newsize)
        #print(im.size)
    #    im.show()
        im.save(folder + "Image" + str(i) + ".jpg")

    t, r, c, ch = d_test.shape 

    folder = res_real[m]
    print("**********************Saving real images at " + folder + "*******************")
    for i in range(t):
        A = (255 * d_test[i].reshape(act_size, act_size, 3)).astype(np.uint8)
        im = Image.fromarray(A)
        newsize = (224, 224) 
    #    im = im.resize(newsize) 
        im.save(folder + "Image" + str(i) + ".jpg")


    folder = res_disp[m]
    print("**********************Saving disparity maps at " + folder + "*******************")
    for i in range(t):
        A = (255 * heatmaps[i].reshape(act_size, act_size, 3)).astype(np.uint8)
        print("MSE: " + str(255*np.mean(heatmaps[i] * heatmaps[i])))
        im = Image.fromarray(A)
        newsize = (224, 224) 
    #    im = im.resize(newsize) 
        im.save(folder + "Image" + str(i) + ".jpg")

# [Optional] Uncomment these line to plot results
#    n = 9
#    plt.figure(figsize=(20, 4))
#    for i in range(1, n + 1):
#        # Display original
#        ax = plt.subplot(2, n, i)
#        plt.gray()
#        plt.imshow(d_test[i].reshape(act_size, act_size, 3))
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)

        
        # Display reconstruction
#        ax = plt.subplot(2, n, i + n)
#        plt.gray()
#        plt.imshow(d_imgs[i].reshape(act_size, act_size, 3))
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#    plt.show()
