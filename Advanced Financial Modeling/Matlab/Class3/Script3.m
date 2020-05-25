[labels, scores]=predict(trainedClassifier.ClassificationSVM,predictors);
cm = confusionmat(response,labels);
ScoreSVMModel=fitPosterior(trainedClassifier.ClassificationSVM,predictors,response);
[labels1,scores1]=predict(ScoreSVMModel,predictors);
