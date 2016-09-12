function Output = SkVM_Gibbs_Block_Classification(modelSuffStats,yyTrain,xxTrain,yyTest,xxTest)
%  input ===================================================================
%   modelSuffStats: contain model sufficient statistic of P, Q
%   yyTrain: label of trainning data in block b[NbTrain x 1]
%   xxTrain: feature of training data in block b[NbTrain x dd]
%   yyTest: label of testing data [NTest x 1]
%   xxTest: feature of testing data [NTest x dd]
%   =========================================================================
%   output ==================================================================
%   Output.AccuracyTrain = classification accuracy on training set
%   Output.AccuracyTest  = classification accuracy on testing set
%   Output.predLabelTest = predicted label : [NTest x 1]

NbTrain=size(xxTrain,1);
dd=size(xxTrain,2);

tic;

%% iteratively sampling
if isempty(modelSuffStats)
    PP=speye(dd);
    %KK=length(unique(yyTrain));
    KK=max(yyTrain);
    QQ=zeros(dd,KK);
    Q0=zeros(dd,1);
    modelSuffStats.NumTrain=0;
    modelSuffStats.trainTime=0;
    weight=histc(yyTrain,1:KK)/NbTrain;
    ww=ones(dd,KK);

else
    Output.NumTrain=modelSuffStats.NumTrain+NbTrain; 
    maxClass=max(yyTrain);
    KK=modelSuffStats.SufficientStatistic.KK;
    PP=modelSuffStats.SufficientStatistic.PP;
    QQ=modelSuffStats.SufficientStatistic.QQ;
    Q0=modelSuffStats.SufficientStatistic.Q0;
    if maxClass>KK
        moreClass=maxClass-KK;
        KK=maxClass;
        for kk=1:moreClass
            QQ(:,end+1)=Q0;
        end
        weight=[modelSuffStats.SufficientStatistic.weight*modelSuffStats.NumTrain/Output.NumTrain; zeros(moreClass,1)];
        weight=weight+histc(yyTrain,1:KK)/Output.NumTrain;
    else
        weight=modelSuffStats.SufficientStatistic.weight*modelSuffStats.NumTrain/Output.NumTrain+histc(yyTrain,1:KK)/Output.NumTrain;
    end
    
    ww=modelSuffStats.SufficientStatistic.ww;

    
end


for kk=1:KK
    idx=find(yyTrain==kk);
    xxTrain(idx,:)=xxTrain(idx,:)*(1-weight(kk));
end

ww=ones(dd,KK);


%% sampling lambda
lambda=zeros(NbTrain,1);
invlambda=zeros(NbTrain,1);
for ii=1:NbTrain
    kk=yyTrain(ii);
    mu_post2=1/(abs(1-xxTrain(ii,:)*ww(:,kk)));
    invlambda(ii)=sample_inverseGaussian(mu_post2,1);
    lambda(ii)=1/invlambda(ii);
    %     if lambda(ii)<0
    %         disp('debug');
    %         break;
    %     end
end

% temp=1./abs(1-sum(xxTrain,2));
% invlambda=sample_inverseGaussian(temp,1);


Lk=-1*ones(NbTrain,1);


Q0=Q0+xxTrain'*Lk;

% tempQQ=xxTrain.*((1+invlambda)*ones(1,dd));
tempQQ = bsxfun(@times, xxTrain, 1+invlambda);

for kk=1:KK
    idx=find(yyTrain==kk);
    Lk=-1*ones(NbTrain,1);
    
    Lk(yyTrain==kk)=1;
    %QQ(:,kk)=QQ(:,kk)+xxTrain'*Lk;
    QQ(:,kk)=QQ(:,kk)+tempQQ'*Lk;
end


bufferMax=100001;
if NbTrain>bufferMax
    nPatch=ceil(NbTrain/bufferMax);
    for ii=1:nPatch-1
        to=ii*bufferMax;
        afrom=(ii-1)*bufferMax+1;
        xx_patch=xxTrain(afrom:to,:);
        %lambda_block=sample_PolyaGamma_approximation(sum(xx_patch,2));
        invlambda_block=invlambda(afrom:to);
        temp = bsxfun(@times, xx_patch, invlambda_block);
        PP = PP+temp'*xx_patch;
    end
    afrom=(nPatch-1)*bufferMax+1;
    xx_patch=xxTrain(afrom:end,:);
    
    %lambda_block=sample_PolyaGamma_approximation(sum(xx_patch,2));
    invlambda_block=invlambda(afrom:end);
    
    temp = bsxfun(@times, xx_patch, invlambda_block);
    PP = PP+temp'*xx_patch;
    
else
    %lambda=sample_PolyaGamma_approximation(sum(xxTrain,2));
    %temp = bsxfun(@times, xxTrain, lambda);
    temp = bsxfun(@times, xxTrain, invlambda);
    PP = PP+temp'*xxTrain;
end



% for kk=1:KK
%     ww(:,kk)=PP\QQ(:,kk);
% end



trainTime=toc;

% PP = inverse_sigma_post
% QQ = sumtempmu

if nargin>3
    nTest=length(yyTest);
    for kk=1:KK
        ww(:,kk)=PP\QQ(:,kk);
    end
    
    %% calculate loss on Test set
    outcome=xxTest*ww;
    [~, predicted_yyTest]=max(outcome,[],2);
    AccuracyTest=100*length(find(predicted_yyTest==yyTest))/nTest;
    Output.AccuracyTest=AccuracyTest;
    Output.SufficientStatistic.predyyTest=predicted_yyTest;
end

Output.SufficientStatistic.weight=weight;
Output.NumTrain=modelSuffStats.NumTrain+NbTrain;
Output.NumFeature=dd;
Output.trainTime=modelSuffStats.trainTime+trainTime;
Output.SufficientStatistic.ww=ww;
Output.SufficientStatistic.PP=PP;
Output.SufficientStatistic.KK=KK;
Output.SufficientStatistic.QQ=QQ;
Output.SufficientStatistic.Q0=Q0;
end

function sample = sample_inverseGaussian( mu, lambda )
% generate sample from inverse Gaussian distribution

%sample from a normal distribution with a mean of 0 and 1 standard deviation
v=randn(1);

y=v*v;

x= mu +(mu*mu*y)/(2*lambda)-sqrt(4*mu*lambda*y+mu*mu*y*y)*(mu/(2*lambda));
test=rand();

if test<=(mu/(mu+x))
    sample =x;
    return;
else
    sample=mu*mu/x;
    return;
end

end
