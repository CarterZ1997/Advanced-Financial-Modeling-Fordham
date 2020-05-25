ScriptHW
X=T(:,2:7);
Y=T(:,9);
ens_boost = fitcensemble(X,Y,'Method','AdaBoostM1');
labels_boost = predict(ens_boost,X);
%%%%%
Z=num2cell(Y);
ZZ=cell2mat(Z);
confusionmat(ZZ,labels_boost)
view(ens_boost.Trained{1,1},'Mode','graph')