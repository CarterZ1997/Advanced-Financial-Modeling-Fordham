rng(438);
newdata2 = datasample(CR2(1:3802, :), 400);
pred = TRAIN.predictFcn(newdata2(:, 2:7));



ninthcol = newdata2(:,9);
confusionmat(ninthcol, pred)