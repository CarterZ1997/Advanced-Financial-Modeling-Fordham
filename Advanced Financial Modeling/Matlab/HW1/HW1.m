% 338
rng(338);
n = 300;
newdata = datasample(CR2(1:3802, :), 300);
% newdata(:, 9) = 1;
joined = [newdata; CR2(3803:3932, :)];

