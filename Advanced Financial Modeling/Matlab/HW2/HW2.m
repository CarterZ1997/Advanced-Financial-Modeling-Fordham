% 338
rng(338);
n = 300;
newdata = datasample(CR2(1:3802, :), 300);
% newdata(:, 9) = 1;
joined = [newdata; CR2(3803:3932, :)];

Ys = joined(:, 9);
Industry = joined(:, 7);
data = [Ys Industry];


leftcounter = 0;
rightcounter = 0;
leftpos = 0;
leftneg = 0;
rightpos = 0;
rightneg = 0;
ginis = zeros(11,1);

for j = 1:11
    for i = 1:430
        if data(i, 2) < (j+0.5)
            leftcounter = leftcounter + 1;
            if data(i, 1) == 1
                leftpos = leftpos + 1;
            else 
                leftneg = leftneg + 1;
            end
        else
            rightcounter = rightcounter + 1;
            if data(i, 1) == 1
                rightpos = rightpos + 1;
            else
                rightneg = rightneg + 1;
            end
        end
    end
    first = (leftcounter / 430) * (1 - (leftpos/leftcounter)^2 - (leftneg/leftcounter)^2);
    second = (rightcounter / 430) * (1 - (rightpos/rightcounter)^2 - (rightneg/rightcounter)^2);
    result = first + second;
    ginis(j) = result;
end        
ginis
