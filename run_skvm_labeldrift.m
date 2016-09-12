% run skvm labeldrift on small fraction of MNIST dataset
clear all;
close all;
addpath(genpath(pwd));

nBlock=10; 
path='sample_data\mnist_labeldrift\';

[yyTest, xxTest]=libsvmread('sample_data\mnist_test');

if isempty(yyTest) || isempty(xxTest)
    disp('fail to load test data');
    return;
end

dd=784;

xxTest=full(xxTest);

[NTest, ddTest]=size(xxTest);

for ii=ddTest+1:dd
    xxTest=[xxTest zeros(NTest,1)];
end

yyTest=yyTest+1;


SKVM.Elapse=[];
SKVM.AccuracyTest=[];
SKVM.F1=[];
OutputSkVM=[];
disp('=================SkVM LabelDrift================');

%pass through each block, then update the model statistic (OutputSLR)
for ii=1:nBlock-1
    mytime=tic;
    
    str=sprintf('%s\\block%d',path,ii-1);
 
    [yyblock, xxblock]=libsvmread(str);
    
    xxblock=full(xxblock);
    
    yyblock=yyblock+1;
    
    dd2=size(xxblock,2);
    for tt=dd2+1:dd
        xxblock=[xxblock zeros(length(yyblock),1)];
    end
      
    OutputSkVM = SkVM_Gibbs_Block_Classification(OutputSkVM,yyblock,full(xxblock),yyTest,full(xxTest));    
 
    ttt=toc(mytime);
    
    SKVM.Elapse=[SKVM.Elapse ttt];
    SKVM.AccuracyTest=[SKVM.AccuracyTest OutputSkVM.AccuracyTest];

    fprintf('SkVM LabelDrift \tBlock %d Acc=%.2f Total Elapse=%.2f (sec)\n',ii,OutputSkVM.AccuracyTest,sum(SKVM.Elapse));
end

%% last iteration with testing
mytime=tic;
    
str=sprintf('%s\\block%d',path,ii);

[yyblock, xxblock]=libsvmread(str);
    
xxblock=full(xxblock);
yyblock=yyblock+1;

    
dd2=size(xxblock,2);
for tt=dd2+1:dd
    xxblock=[xxblock zeros(length(yyblock),1)];
end

OutputSkVM = SkVM_Gibbs_Block_Classification(OutputSkVM,yyblock,full(xxblock),yyTest,full(xxTest));
ttt=toc(mytime);

SKVM.Elapse=[SKVM.Elapse ttt];
SKVM.AccuracyTest=[SKVM.AccuracyTest OutputSkVM.AccuracyTest];

fprintf('SkVM LabelDrift \tBlock %d Total Elapse=%.2f (sec) \tAcc=%.2f\n',ii+1,sum(SKVM.Elapse),OutputSkVM.AccuracyTest);


%% plot
plot(cumsum(SKVM.Elapse),SKVM.AccuracyTest,'-sr','LineWidth',2);
xlabel('Running Time','fontsize',16);
ylabel('Accuracy','fontsize',16);
title('SkVM LabelDrift Classification','fontsize',16);
