clc
clear all
close all

path = 'Training';

data = imageDatastore(path,...
	'IncludeSubfolders',true,...
    'LabelSource','foldernames');

[merchImagesTrain,merchImagesTest] = splitEachLabel(data,0.8,'randomized');

merchImagesTrain = shuffle(merchImagesTrain);

inputSize = [224 224 3];

numClasses = 13;
         
newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','loss3-classifier', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
     
newClassLayer = classificationLayer('Name','output');
         
       
layers = [...
    imageInputLayer(inputSize,'Name','input')
    convolution2dLayer(3,16,'Padding','same','Name','conv1')
    reluLayer('Name','relu1')
    maxPooling2dLayer([2 2],'Padding','same','Name','mp1')
	convolution2dLayer(3,8,'Padding','same','Name','conv2')
    reluLayer('Name','relu2')
    maxPooling2dLayer([2 2],'Padding','same','Name','mp2')
	convolution2dLayer(3,8,'Padding','same','Name','conv3')
    reluLayer('Name','relu3')
    maxPooling2dLayer([2 2],'Padding','same','Name','mp3')
    newLearnableLayer
    softmaxLayer('Name','sm1')
    newClassLayer];

lgraph = layerGraph(layers);
        
options = trainingOptions('adam', ...
	'InitialLearnRate',0.0001, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',merchImagesTest, ...
    'ValidationFrequency',10, ...
    'MiniBatchSize',1024, ...
    'Verbose',true, ...
    'Plots','training-progress');

classifier2S = trainNetwork(merchImagesTrain,lgraph,options);