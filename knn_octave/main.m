%% Initialization
clear ; close all;

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

X_train = loadMNISTImages('train-images-idx3-ubyte');
y_train = loadMNISTLabels('train-labels-idx1-ubyte');
%X_train = X_train(:,1:10000);
y_train = y_train';
%y_train = y_train(:,1:10000);

X_test = loadMNISTImages('t10k-images-idx3-ubyte');
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
y_test = y_test';
%X_test = X_test(:,1:1000);
%y_test = y_test(:,1:1000);


%setup parameters
K = 100;

predict = knn(X_test, X_train, y_train, K);

fprintf('\nAccuracy: %f\n', mean(double(predict == y_test)) * 100);

