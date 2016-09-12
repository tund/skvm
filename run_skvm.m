% run skvm on small fraction of MNIST dataset
clear all;
close all;
addpath(genpath(pwd));

[yyTrain, xxTrain]=libsvmread('sample_data\mnist_train');
[yyTest, xxTest]=libsvmread('sample_data\mnist_test');

if isempty(yyTest) || isempty(xxTest)
    disp('fail to load test data');
    return;
end

dd=784;

xxTest=full(xxTest);
xxTrain=full(xxTrain);

[NTest, ddTest]=size(xxTest);
[NTrain, ddTrain]=size(xxTrain);

for ii=ddTest+1:dd
    xxTest=[xxTest zeros(NTest,1)];
end

for ii=ddTrain+1:dd
    xxTrain=[xxTrain zeros(NTrain,1)];
end

yyTest=yyTest+1;
yyTrain=yyTrain+1;

disp('=================SkVM================');

mytime=tic;

OutputSkVM = SkVM_Gibbs_Block_Classification([],yyTrain,xxTrain,yyTest,xxTest);

ttt=toc(mytime);

fprintf('SkVM Classification on (tiny) MNIST \tBlock %d \tTotal Elapse=%.3f (sec) \tAcc=%.2f\n',ii+1,ttt,OutputSkVM.AccuracyTest);

