
% 338
rng(338);
newdata = datasample(CR2(1:3802, :), 300);
joined = [newdata; CR2(3803:3932, :)];
rng(438);
testdata = datasample(CR2(1:3802, :), 400);
testdata = [testdata; CR2(3803:3932, :)];

% Simple Decision Tree
% The output model is SimpleTree, and use SimpleTree.predictFcn(X) to pred
% view(SimpleTree.ClassificationTree, 'Mode', 'Graph')


% Optimized Decision Tree, use OptTree.predictFcn(X) to pred
% optT_pred = SimpleTree.ClassificationTree.X;
% optT_resp = SimpleTree.ClassificationTree.Y;
% OptTree = fitctree(optT_pred, optT_resp, 'OptimizeHyperparameters', 'auto');
% view(OptTree, 'Mode', 'Graph')


% Bagged Tree Ensemble, use ens.predictFcn(X) to pred
% bagX = joined(:,2:7);
% bagY = joined(:,9);
% ens = fitcensemble(bagX, bagY, 'Method', 'Bag');
% [label_ens,score_score_ens] = oobPredict(ens);
% bagZ=num2cell(bagY);
% bagZZ=cell2mat(bagZ);
% confusionmat(bagZZ,label_ens)
% view(ens.Trained{1,1},'Mode','graph')


% Boosted Tree Ensemble, use ens.boost.predictFcn(X) to pred
% boostX = joined(:,2:7);
% boostY = joined(:,9);
% ens_boost = fitcensemble(boostX, boostY, 'Method', 'AdaBoostM1');
% labels_boost = predict(ens_boost, boostX);
% boostZ=num2cell(boostY);
% boostZZ=cell2mat(boostZ);
% confusionmat(boostZZ, labels_boost)
% view(ens_boost.Trained{1,1},'Mode','graph')


% Random Forest, use RFor1.predictFcn(X) to pred
% rfX = joined(:,2:7);
% rfY = joined(:,9);
% rng(1); % For reproducibility
% RFor1 = TreeBagger(20, rfX, rfY, 'Method','classification',...
%     'OOBPrediction','on','NumPredictorsToSample',4);
% [label_rf,score_score_rf] = oobPredict(RFor1);
% rfZ = num2cell(rfY);
% rfZZ=cell2mat(rfZ);
% Q=str2double(label_rf);
% confusionmat(rfZZ,Q)
% view(RFor1.Trees{1},'Mode','graph')

% Testing:
X_test = testdata(:,2:7);
Y_test = testdata(:,9);

%Simple Decision Tree
% label_Simple_Tree = predict(SimpleTree.ClassificationTree,X_test);
% confusionmat(Y_test,label_Simple_Tree)

%Opt Tree
% label_opt = predict(OptTree,X_test);
% confusionmat(Y_test,label_opt)

%Bagged
% label_bag = predict(ens,X_test);
% confusionmat(Y_test,label_bag)

%RF
% label_RF = predict(RFor1,X_test);
% label_rf = str2double(label_RF);
% confusionmat(Y_test,label_rf)

%Boosted
% label_boost = predict(ens_boost,X_test);
% confusionmat(Y_test,label_boost)
