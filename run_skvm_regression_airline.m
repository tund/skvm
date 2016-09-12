%% skvm regression on a tiny fraction of Airline dataset
clear all;
close all;
addpath(genpath(pwd));

[yyTrain, xxTrain]=libsvmread('sample_data\airlines_regression_train');
[yyTest, xxTest]=libsvmread('sample_data\airlines_regression_test');

if isempty(yyTest) || isempty(xxTest)
    disp('fail to load test data');
    return;
end

maxDistance=4983.0;

xxTrain(:,end)=xxTrain(:,end)/maxDistance;
xxTest(:,end)=xxTest(:,end)/maxDistance;

disp('=================SkVM Regression================');

mytime=tic;

OutputSkVM = SkVM_Gibbs_Block_Regression([],yyTrain,xxTrain,yyTest,xxTest);

ttt=toc(mytime);    


fprintf('SkVM Regression on (tiny) Airline dataset \tTotal Elapse=%.3f (sec) \tRMSE=%.2f MAE=%.2f\n',...
    ttt,OutputSkVM.RMSETest,OutputSkVM.MAETest);

