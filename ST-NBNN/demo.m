close all;
clear all;
clc;

addpath(genpath('./lib/liblinear'));
addpath(genpath('./cvx-w64/cvx'));


for i = 1:11
    load(['./MHAD/data_' num2str(i-1) '.mat'], 'train_X_list', 'train_Y', 'test_X_list', 'test_Y');
    trainX{i} = train_X_list;
    testX{i}  = test_X_list;
end


penalty = 1;
Lamdas  = 1;
weight  = 10;


[accNBNNTrain accNBNNTest accTrain accTest] = stm_train(trainX, train_Y, testX, test_Y, Lamdas, penalty,  weight);
fprintf(' (NBNN -> S-NBNN -> ST-NBNN) : (%f) -> (%f) -> (%f)\n', accNBNNTest, accTest(1), accTest(2));

