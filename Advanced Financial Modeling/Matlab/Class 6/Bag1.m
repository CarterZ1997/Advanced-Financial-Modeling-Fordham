ScriptHW
X=T(:,2:7);
Y=T(:,9);
ens = fitcensemble(X,Y,'Method','Bag');
[label_ens,score_score_ens] = oobPredict(ens);
%%%%%
Z=num2cell(Y);
ZZ=cell2mat(Z);
confusionmat(ZZ,label_ens)
view(ens.Trained{1,1},'Mode','graph')
