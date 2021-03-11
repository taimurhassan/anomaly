# Unsupervised Anomaly Instance Segmentation for Baggage Threat Recognition

Configurations:
1) Please install the following libraries, or load the provided ‘environment.yml’ file

  a. TensorFlow-GPU 2.3.1
  b. Keras-GPU 2.3.1
  c. OpenCV 4.2
  d. Imgaug
  e. Tqdm
We used Anaconda with Python 3.7.8 for simulations. Also please install MATLAB R2020b as well
with deep learning and computer vision toolboxes.

2) Load the desired dataset in the dataset folder within ‘code and results’ folder. The dataset has
following hierarchy:

-datasets
  -dataset name (e.g. sixray)
    -abnormal
    -input
    -normal (only for SIXray for one-time training)
    -results
      -disp
      -fake
      -real
      -results

For the one-time training, please provide the normal samples in ‘normal’ folder within SIXray
folder.
For the test datasets, please provide test (abnormal) samples in ‘input’ folder for all datasets.

3) Please run the ‘gwfs.m’ in MATLAB for stylization, the stylized output is saved in the ‘abnormal’
folder. Please note that this step is only to be performed for the test scans.

4) Afterward, please run the ‘main.py’ to produce reconstructions and disparity maps:

python main.py

We provided the trained encoder-decoder model that can be used for reconstruction. However,
for training, please check to the flag at Line 58 within config.py to ‘True’.

5) After completing step 4, please run ‘cluster.m’ file to produce final outputs. The outputs are
saved in the ‘results->results folder’ for each dataset.
