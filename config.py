import tensorflow.keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras import backend as K
   
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

from tensorflow.keras import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras import models
import tensorflow as tf
from model import ProposedModel, getAssembledModel

numDatasets = 5

tr_folder = "datasets/sixray/normal/" # path to training dataset

te_folders = [] # path to test datasets
te_folders.append("datasets/sixray/abnormal/") 
te_folders.append("datasets/gdxray/abnormal/") 
te_folders.append("datasets/opixray/abnormal/") 
te_folders.append("datasets/compass/abnormal/") 
te_folders.append("datasets/mvtec/abnormal/") 

res_fake = [] # path to store reconstructed images for each dataset
res_fake.append("datasets/sixray/results/fake/")
res_fake.append("datasets/gdxray/results/fake/")
res_fake.append("datasets/opixray/results/fake/")
res_fake.append("datasets/compass/results/fake/")
res_fake.append("datasets/mvtec/results/fake/")

res_real = [] # path to store real images for each dataset
res_real.append("datasets/sixray/results/real/")
res_real.append("datasets/gdxray/results/real/")
res_real.append("datasets/opixray/results/real/")
res_real.append("datasets/compass/results/real/")
res_real.append("datasets/mvtec/results/real/")

res_disp = [] # path to store disparity maps for each dataset
res_disp.append("datasets/sixray/results/disp/")
res_disp.append("datasets/gdxray/results/disp/")
res_disp.append("datasets/opixray/results/disp/")
res_disp.append("datasets/compass/results/disp/")
res_disp.append("datasets/mvtec/results/disp/")

x_train = [] # contains training patches
x_test = [] # contains test patches
x_valid = [] # contains validation patches

i_i = [] # contains indices (used for patch stitching)
i_j = [] # contains indices (used for patch stitching)

doTraining = False

act_size = 2240 # image size
p = 224 # patch size