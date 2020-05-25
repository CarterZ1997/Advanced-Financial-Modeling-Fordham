
c = cvpartition(160,'KFold',5);
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus');
svmmod = fitcsvm(predictors,response,'KernelFunction','rbf','Standardize',true,...
    'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',opts)
L=resubLoss(svmmod);
accuracy_opt=1-L;

[label,score] = predict(svmmod,predictors);
cm_opt=confusionmat(response,label);
