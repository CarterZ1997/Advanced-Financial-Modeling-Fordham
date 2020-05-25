ScriptHW
X=T(:,2:7);
Y=T(:,8);

rng(1); % For reproducibility
RFor1 = TreeBagger(20,X,Y,'Method','classification',...
    'OOBPrediction','on','NumPredictorsToSample',4);
[label_rf,score_score_rf] = oobPredict(RFor1);

Z=table2cell(Y);
ZZ=cell2mat(Z);
Q=str2double(label_rf);


confusionmat(ZZ,Q)
view(RFor1.Trees{1},'Mode','graph')
