# Unsupervised Anomaly Instance Segmentation for Baggage Threat Recognition

## Introduction
This repository contains the implementation of the proposed unsupervised anomaly instance segmentation framework, as shown below: 

![TP](/images/anomaly.png)

[Paper Link](https://arxiv.org/abs/2107.07333)

## Configurations
Please install the following libraries, or load the provided ‘environment.yml’ file

1. TensorFlow-GPU 2.3.1
2. Keras-GPU 2.3.1
3. OpenCV 4.2
4. Imgaug
5. Tqdm
  
We used Anaconda with Python 3.7.8 for simulations. Also please install MATLAB R2020b as well
with deep learning and computer vision toolboxes.

## Datasets
The X-ray datasets can be downloaded from the following URLs: 
1. [GDXray](https://domingomery.ing.puc.cl/material/gdxray/) 
2. [SIXray](https://github.com/MeioJane/SIXray) 
3. [OPIXray](https://github.com/OPIXray-author/OPIXray)
4. [COMPASS-XP](https://figshare.com/articles/dataset/COMPASS-XP/9249791/1)

## Steps 

1. Load the desired dataset in the dataset folder. The dataset hierarchy is as follows:

```
├── datasets
│   ├── dataset name (e.g. sixray)
│   │   └── abnormal
│   │   └── input
│   │   └── normal (only contained in the SIXray for one-time training)
│   │   └── results
│   │   │   └── disp
│   │   │   └── fake
│   │   │   └── real
│   │   │   └── results
```

2) For the one-time training, please provide the normal samples in ‘normal’ folder within SIXray folder.

3) For the test datasets, please provide test (abnormal) samples in ‘input’ folder for all datasets.

4) Please run the ‘gwfs.m’ in MATLAB for stylization, the stylized output is saved in the ‘abnormal’
folder. Please note that this step is only to be performed for the test scans.

5) Afterward, please run the ‘main.py’ to produce reconstructions and disparity maps:

```
python main.py
```

6) We provided the trained encoder-decoder model that can be used for reconstruction. However, for training, please check to the flag at Line 58 within config.py to ‘True’.

7) After completing step 6, please run ‘cluster.m’ file to produce final outputs. The outputs are saved in the ‘results->results folder’ for each dataset.


## Results
The results of the proposed framework (for each dataset) are presented in the '…/datasets/(dataset-name)/results' folder. 

## Citation
If you use the proposed framework (or any part of this code in your research), please cite the following paper:

```
@inproceedings{Hassan2021Anomaly,
  title   = {Unsupervised Anomaly Instance Segmentation for Baggage Threat Recognition},
  author  = {Taimur Hassan and Samet Akcay and Mohammed Bennamoun and Salman Khan and Naoufel Werghi},
  note = {Journal of Ambient Intelligence and Humanized Computing},
  year = {2021}
}
```

## Contact
If you have any query, please feel free to contact us at: taimur.hassan@ku.ac.ae.
