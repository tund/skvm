function Output = SkVM_Gibbs_Block_Regression(modelSuffStats,yyTrain,xxTrain,yyTest,xxTest)
%  input ===================================================================
%   modelSuffStats: contain model sufficient statistic of P, Q
%   yyTrain: label of trainning data in block b[NbTrain x 1]
%   xxTrain: feature of training data in block b[NbTrain x dd]
%   yyTest: label of testing data [NTest x 1]
%   xxTest: feature of testing data [NTest x dd]
%   =========================================================================
%   output ==================================================================
%   Output.RMSETest = Root Mean Squared Error on testing set
%   Output.MAETest  = Mean Absolute Error on testing set
%   Output.SufficientStatistic.predyyTest = predicted outcome : [NTest x 1]

NbTrain=size(xxTrain,1);
dd=size(xxTrain,2);

epsilon=0.5;
tic;

%% iteratively sampling
if isempty(modelSuffStats)
    PP=speye(dd);
    %KK=length(unique(yyTrain));
    %KK=max(yyTrain);
    QQ=zeros(dd,1);
    modelSuffStats.NumTrain=0;
    modelSuffStats.trainTime=0;
    %weight=histc(yyTrain,1:KK)/NbTrain;
    ww=ones(dd,1);

else
    Output.NumTrain=modelSuffStats.NumTrain+NbTrain; 
    PP=modelSuffStats.SufficientStatistic.PP;
    QQ=modelSuffStats.SufficientStatistic.QQ;
  
    ww=modelSuffStats.SufficientStatistic.ww;
end


%% sampling lambda

bufferMax=100001;
if NbTrain>bufferMax
    nPatch=ceil(NbTrain/bufferMax);
    for ii=1:nPatch-1
        to=ii*bufferMax;
        afrom=(ii-1)*bufferMax+1;
        xx_patch=xxTrain(afrom:to,:);
        yy_patch=yyTrain(afrom:to);
        
        mu_post2=1.0./double(abs(yy_patch-xx_patch*ww-epsilon));
        invlambda_block1=sample_inverseGaussianVector(mu_post2,1);
        invlambda_block1(invlambda_block1<0.00001)=0.00001;

        mu_post2=1.0./double(abs(yy_patch-xx_patch*ww+epsilon));
        invlambda_block2=sample_inverseGaussianVector(mu_post2,1);
        invlambda_block2(invlambda_block2<0.00001)=0.00001;

        temp = bsxfun(@times, xx_patch, invlambda_block1+invlambda_block2);
        PP = PP+temp'*xx_patch;
    end
    afrom=(nPatch-1)*bufferMax+1;
    xx_patch=xxTrain(afrom:end,:);
    yy_patch=yyTrain(afrom:end);
    
    mu_post2=1.0./double(abs(yy_patch-xx_patch*ww-epsilon));
    invlambda_block1=sample_inverseGaussianVector(mu_post2,1);
    invlambda_block1(invlambda_block1<0.00001)=0.00001;
    
    mu_post2=1.0./double(abs(yy_patch-xx_patch*ww+epsilon));
    invlambda_block2=sample_inverseGaussianVector(mu_post2,1);
    invlambda_block2(invlambda_block2<0.00001)=0.00001;
    
    temp = bsxfun(@times, xx_patch, invlambda_block1+invlambda_block2);
    PP = PP+temp'*xx_patch;
    
    tempsigma=(double(yy_patch-epsilon).*invlambda_block1+...
        double(yy_patch+epsilon).*invlambda_block2)*ones(1,dd).*xx_patch;
    QQ=QQ+sum(tempsigma)';

else

    mu_post2=1.0./double(abs(yyTrain-xxTrain*ww-epsilon));
    invlambda_block1=sample_inverseGaussianVector(mu_post2,1);
    invlambda_block1(invlambda_block1<0.00001)=0.00001;
    
    mu_post2=1.0./double(abs(yyTrain-xxTrain*ww+epsilon));
    invlambda_block2=sample_inverseGaussianVector(mu_post2,1);
    invlambda_block2(invlambda_block2<0.00001)=0.00001;
    
    temp = bsxfun(@times, xxTrain, invlambda_block1+invlambda_block2);
    PP = PP+temp'*xxTrain;
    
    tempsigma=(double(yyTrain-epsilon).*invlambda_block1+...
        double(yyTrain+epsilon).*invlambda_block2)*ones(1,dd).*xxTrain;
    QQ=QQ+sum(tempsigma)';
end



trainTime=toc;


if nargin>3
    nTest=length(yyTest);
    ww=PP\QQ;
    
    %% calculate loss on Test set
    outcome=xxTest*ww;
    SquaredError=(outcome-yyTest).^2;
    RMSETest=sqrt(mean(SquaredError));
    MAETest=mean(abs(outcome-yyTest));
    
    %[~, predicted_yyTest]=max(outcome,[],2);
    %AccuracyTest=100*length(find(predicted_yyTest==yyTest))/nTest;
    Output.RMSETest=RMSETest;
    Output.MAETest=MAETest;
    Output.SufficientStatistic.predyyTest=outcome;
end

Output.NumTrain=modelSuffStats.NumTrain+NbTrain;
Output.NumFeature=dd;
Output.trainTime=modelSuffStats.trainTime+trainTime;
Output.SufficientStatistic.ww=ww;
Output.SufficientStatistic.PP=PP;
Output.SufficientStatistic.QQ=QQ;
end
