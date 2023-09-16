% Ensemble Learning 
clear all
close all
clc
%% Model 1
% Training set input
rootFolder = fullfile('F:\ICM trainingset');
categories  = {'real','attack'};
trainingset1 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
tbl = countEachLabel(trainingset1);
%% Input for development set 
rootFolder = fullfile('F:\CM testingset');
categories  = {'real','attack'};
developmentset1 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames'); 
%% Input for Testing set 
rootFolder = fullfile('F:\OULU testing set');
categories = {'real','attack'};
testingsetdata1 = imageDatastore(fullfile(rootFolder, categories),  'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Extracting  labels for training, development, and test set
trainingLabels = trainingset1.Labels;
developmentlabel = developmentset1.Labels;
testinglabel = testingsetdata1.Labels;
%% Input for Deep learning model
net = densenet201;
%% fine tunning
net.Layers(1)
inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(height(tbl), ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
augimdsTrain1 = augmentedImageDatastore(inputSize(1:2),trainingset1);
developmentset1.ReadFcn = @(filename)readAndPreprocessImage(filename);
miniBatchSize = 32;
options = trainingOptions('sgdm', ...
     'ExecutionEnvironment','gpu', ... 
         'MiniBatchSize',miniBatchSize, ...
          'Shuffle','every-epoch', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{developmentset1,developmentlabel}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,1));

net1 = trainNetwork(augimdsTrain1,lgraph,options); 

%% Features extraction based on the last average pooling layer of densenet-201 for the training set 
trainingset1.ReadFcn = @(filename)readAndPreprocessImage(filename);
featureLayer =   'avg_pool';
trainingFeatures1 = activations(net1, trainingset1, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of densenet-201 for the development set 
developmentset1.ReadFcn = @(filename)readAndPreprocessImage(filename);
developmentFeatures1 = activations(net1,developmentset1, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of densenet-201 for the testing set 
testingsetdata1.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures1 = activations(net1, testingsetdata1, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%%
% Recurrent neural network 1
% Converting data into LSTM FORMAT
rng(13); % For reproducibility
trainf = {};
trainf{end+1} =  trainingFeatures1;

trainlabl = {};
trainlabl{end+1} = trainingLabels';

train1 = {};
train1{end+1} = developmentFeatures1;
% 
train2 = {};
train2{end+1} = developmentlabel';


numFeatures = 1920;
 numHiddenUnits =500;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
         lstmLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

RNN1 = trainNetwork(trainf',trainlabl,layers,options);

[~, devlpscores1] = classify(RNN1, developmentFeatures1);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores1 =devlpscores1';

 [~,~,Info]=vl_roc(numericLabels1,devlpscores1(:,1));
 EER = Info.eer*100;
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [~, test_scores1] = classify(RNN1, testingFeatures1);
 testscores1 = test_scores1';
 numericLabels = grp2idx(testinglabel);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores1(numericLabels==1);
 attack_scores2 =  testscores1(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER1 = (FAR+FRR)/2
[x1,y1,~,AUC1] = perfcurve( numericLabels, testscores1(:,1),1);
AUC1
plot(x1,y1,'-g','LineWidth',2.5,'MarkerSize',2.5)
grid on
hold on
%  
 %%  Model 2
 % input for training set
rootFolder = fullfile('F:\IICM_TRAININGSET_0.5');
categories  = {'real','attack'};
trainingset2 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
tbl = countEachLabel(trainingset2);
%% Input for development set 
rootFolder = fullfile('F:\ICM testingset');
categories  = {'real','attack'};
developmentset2 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames'); 
%% Input for Testing set 
rootFolder = fullfile('F:\Oulu testing set');
categories = {'real','attack'};
testingsetdata2 = imageDatastore(fullfile(rootFolder, categories),  'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%%
net.Layers(1)
inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);

[learnableLayer,classLayer] = findLayersToReplace(lgraph);


if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(height(tbl), ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
augimdsTrain2 = augmentedImageDatastore(inputSize(1:2),trainingset2);
developmentset2.ReadFcn = @(filename)readAndPreprocessImage(filename);

miniBatchSize = 32;
valFrequency = floor(numel(augimdsTrain2.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ... 
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',200, ...
    'InitialLearnRate',0.0001, ...
    'Shuffle','every-epoch', ...   
    'ValidationData', developmentset2, ...
   'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
     'Plots','training-progres', ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,1));
net2 = trainNetwork(augimdsTrain2,lgraph,options); 

trainingLabels = trainingset2.Labels;
%% Features extraction based on the last average pooling layer of densenet-201 for the training set 
trainingset2.ReadFcn = @(filename)readAndPreprocessImage(filename);
featureLayer =  'avg_pool' ;
trainingFeatures2 = activations(net2, trainingset2, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of densenet-201 for the development set 
developmentset2.ReadFcn = @(filename)readAndPreprocessImage(filename);
developmentFeatures2 = activations(net2,developmentset2, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of densenet-201 for the testing set 
testingsetdata2.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures2 = activations(net2, testingsetdata2, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 %%
% Recurrent neural network 2
% Converting data into BiLSTM FORMAT
rng(13); % For reproducibility
trainf = {};
trainf{end+1} =  trainingFeatures2;

trainlabl = {};
trainlabl{end+1} = trainingLabels';

train1 = {};
train1{end+1} = developmentFeatures2;
% 
train2 = {};
train2{end+1} = developmentlabel';


numFeatures = 1920;
 numHiddenUnits =20;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
         bilstmLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

RNN2 = trainNetwork(trainf',trainlabl,layers,options);

[~,  devlpscores2] = classify(RNN2, developmentFeatures2);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores2 = devlpscores2';

 [~,~,Info]=vl_roc(numericLabels1,devlpscores2(:,1));
 EER = Info.eer*100;
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [~, testscores2] = classify(RNN2, testingFeatures2);
 testscores2 = testscores2';
 numericLabels = grp2idx(testinglabel);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores2(numericLabels==1);
 attack_scores2 =  testscores2(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER2 = (FAR+FRR)/2
 
[x2,y2,~,AUC2] = perfcurve( numericLabels, testscores2(:,1),1);
AUC2
 plot(x2,y2,'-m','LineWidth',2.5,'MarkerSize',2.5)
 grid on
 hold on
 
  %% Model 3
rootFolder = fullfile('I'F:\CM_trainingset_1.5');
categories  = {'real','attack'};
trainingset3 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
tbl = countEachLabel(trainingset3);
%% Input for development set 
rootFolder = fullfile('F:\ICM testingsettesting');
categories  = {'real','attack'};
developmentset3= imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames'); 
%% Input for Testing set 
rootFolder = fullfile('F:\Oulu testing set');
categories = {'real','attack'};
testingsetdata3 = imageDatastore(fullfile(rootFolder, categories),  'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%%
net.Layers(1)
inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
 

% numClasses = numel(categories(trainingset.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(height(tbl), ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
augimdsTrain3 = augmentedImageDatastore(inputSize(1:2),trainingset3);
developmentset3.ReadFcn = @(filename)readAndPreprocessImage(filename);
miniBatchSize = 32;
valFrequency = floor(numel(augimdsTrain3.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ... 
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',200, ...
    'InitialLearnRate',0.0001, ...
    'Shuffle','every-epoch', ...   
    'ValidationData', developmentset3, ...
   'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
     'Plots','training-progres', ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,1));
net3 = trainNetwork(augimdsTrain3,lgraph,options); 

%% Extracting  labels for training, development, and test set
trainingLabels = trainingset3.Labels;
%% Features extraction based on the last average pooling layer of ResNet for the training set 
trainingset3.ReadFcn = @(filename)readAndPreprocessImage(filename);
featureLayer =  'avg_pool' ;
trainingFeatures3 = activations(net3, trainingset3, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the development set 
developmentset3.ReadFcn = @(filename)readAndPreprocessImage(filename);
developmentFeatures3 = activations(net3,developmentset3, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the testing set 
testingsetdata3.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures3 = activations(net3, testingsetdata3, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 %%
% Recurrent neural network 3
% Converting data into GRU FORMAT
rng(13);
trainf = {};
trainf{end+1} =  trainingFeatures3;

trainlabl = {};
trainlabl{end+1} = trainingLabels';

train1 = {};
train1{end+1} = developmentFeatures3;
% 
train2 = {};
train2{end+1} = developmentlabel';


numFeatures = 1920;
 numHiddenUnits =20;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
         gruLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

RNN3 = trainNetwork(trainf',trainlabl,layers,options);

[predictedLabels4,  devlpscores3] = classify(RNN3, developmentFeatures3);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores3 = devlpscores3';

 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscores3(:,1));
 EER = Info.eer*100
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [predictedLabels2, testscores3] = classify(RNN3, testingFeatures3);
testscores3 = testscores3';
 numericLabels = grp2idx(testinglabel);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 = testscores3(numericLabels==1);
 attack_scores2 =  testscores3(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER3 = (FAR+FRR)/2
 
[x3,y3,threshold,AUC3] = perfcurve(numericLabels, testscores3(:,1),1);
AUC3
 plot(x3,y3,'-b','LineWidth',2.5,'MarkerSize',2.5)
 grid on
 hold on

 %% 
% stacking ensemble learning
 % META-MODEL LEARNING (GRU)
Developementscore = horzcat(devlpscores1,  devlpscores2, devlpscores3);
finaltestscores  = horzcat(testscores1,  testscores2, testscores3);

rng(1) % For reproducibility
trainf = {};
trainf{end+1} =  Developementscore';
trainlabl = {};
trainlabl{end+1} = developmentlabel';
numFeatures =6;
 numHiddenUnits =100;
numClasses = 2;
layers = [...
    sequenceInputLayer(numFeatures)
   gruLayer(numHiddenUnits)
     fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
maxEpochs = 100;
miniBatchSize = 32;
options = trainingOptions('sgdm', ...
     'ExecutionEnvironment','gpu', ... 
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
      'Plots','training-progress',...
    'Verbose',true);
META = trainNetwork(trainf',trainlabl,layers,options);

[~, devlp_scoresfinal] = classify(META, Developementscore');
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscoresfianl =devlp_scoresfinal';
 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscoresfianl(:,1));
 EER = Info.eer*100;
 threashold = Info.eerThreshold;
 % Evaluation for testing set for HTER 
 [~, test_scoresfinal] = classify(META, finaltestscores');
 testscoresfinal = test_scoresfinal';
 numericLabels = grp2idx(testinglabel);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 = testscoresfinal(numericLabels==1);
 attack_scores2 = testscoresfinal(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 MetaHTER = (FAR+FRR)/2
[x4,y4,threshold, AUC4] = perfcurve(numericLabels, testscoresfinal(:,1),1);
AUC4
plot(x4,y4,'-c','LineWidth',2.5,'MarkerSize',2.5)
grid on
hold off
legend('CNN-LSTM','CNN-BiLSTM','CNN-GRU', 'Meta-Model')

  %%
 % Resize images 
     function Iout = readAndPreprocessImage(filename)

       I = imread(filename);

         if ismatrix(I)
            I = cat(3,I,I,I);
         end
     

           Iout = imresize(I, [224 224]);
            
     end
    %%
 % Function for early stopping
 
 function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end
