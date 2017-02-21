function [ stats, output, e, L ] = main(  )

% Classifier
type = 2;

fprintf('--- Loading data ---\n');
% Load
load('data4students.mat');

% Initialise all params
train_x = datasetInputs{1};
val_x = datasetInputs{2};
test_x = datasetInputs{3};
train_y = datasetTargets{1};
val_y = datasetTargets{2};
test_y = datasetTargets{3};

disp(size(train_x));
disp(size(train_y));

% Key data params
inputSize = size(train_x,2);
outputSize = size(train_y, 2); % in case of classification it should be equal to the number of classes

fprintf('--- Data loaded ---\n');

% ----- START: NN setup stuff -----

% Linear as recommended in CW spec
inputActivationFunction = 'linear'; 

% Normalisation
fprintf('--- Normalise ---\n');

train_x = normaliseData(inputActivationFunction, train_x, []);
val_x = normaliseData(inputActivationFunction, val_x, []);
test_x = normaliseData(inputActivationFunction, test_x, []);

fprintf('--- Normalise complete ---\n');

% Network topography
hiddenActivationFunctions = {'ReLu', 'ReLu', 'ReLu', 'ReLu','softmax'};
hiddenLayers = [500 500 500 1000 outputSize];

% Initialise NN params
nn = paramsNNinit(hiddenLayers, hiddenActivationFunctions);

% Set some NN params
%-----
% This should probably be increased
nn.epochs = 200;

% --- Learning rate ---
% set initial learning rate
nn.trParams.lrParams.initialLR = 0.01; 
% set the threshold after which the learning rate will decrease (if type
% = 1 or 2)
nn.trParams.lrParams.lrEpochThres = 50;
% set the learning rate update policy (check manual)
% 1 = initialLR*lrEpochThres / max(lrEpochThres, T), 2 = scaling, 3 = lr / (1 + currentEpoch/lrEpochThres)
nn.trParams.lrParams.schedulingType = 1;

% --- Momentum ---
% Initial params
nn.trParams.momParams.initialMomentum = 0.5;
nn.trParams.momParams.finalMomentum = 0.99;
nn.trParams.momParams.schedulingType = 1;

%set the epoch where the learning will begin to increase
nn.trParams.momParams.momentumEpochLowerThres = 10;
%set the epoch where the learning will reach its final value (usually 0.9)
nn.trParams.momParams.momentumEpochUpperThres = 15;

% set weight constraints
nn.weightConstraints.weightPenaltyL1 = 0;
nn.weightConstraints.weightPenaltyL2 = 0;
nn.weightConstraints.maxNormConstraint = 4;

% show diagnostics to monnitor training  
nn.diagnostics = 1;
% show diagnostics every "showDiagnostics" epochs
nn.showDiagnostics = 10;

% show training and validation loss plot
nn.showPlot = 1;

% use bernoulli dropout
nn.dropoutParams.dropoutType = 0;

% if 1 then early stopping is used
nn.earlyStopping = 0;
nn.max_fail = 10;

nn.type = type;

% set the type of weight initialisation (check manual for details)
nn.weightInitParams.type = 8;

% set training method
% 1: SGD, 2: SGD with momentum, 3: SGD with nesterov momentum, 4: Adagrad, 5: Adadelta,
% 6: RMSprop, 7: Adam
nn.trainingMethod = 2;
%-----------

% initialise weights
[W, biases] = initWeights(inputSize, nn.weightInitParams, hiddenLayers, hiddenActivationFunctions);

nn.W = W;
nn.biases = biases;

[nn, Lbatch, L_train, L_val, clsfError_train, clsfError_val]  = trainNN(nn, train_x, train_y, val_x, val_y);

nn = prepareNet4Testing(nn);

% % visualise weights of first layer
% figure()
% visualiseHiddenLayerWeights(nn.W{1},visParams.col,visParams.row,visParams.noSubplots);

[stats, output, e, L] = evaluateNNperformance( nn, test_x, test_y);


end

