rng(1);           % For reproducibility
n = 100;         % Number of points per quadrant
r1 = sqrt(rand(2*n,1));       % Random radii
t1 = [pi/2*rand(n,1); (pi/2*rand(n,1)+pi)];  % Random angles for Q1 and Q3
X1 = [r1.*cos(t1) r1.*sin(t1)];     % Polar-to-Cartesian conversion
r2 = sqrt(rand(2*n,1));
t2 = [pi/2*rand(n,1)+pi/2;
(pi/2*rand(n,1)-pi/2)]; %    Random angles for Q2 and Q4
X2 = [r2.*cos(t2) r2.*sin(t2)];
X = [X1; X2]; % Predictors
Y = ones(4*n,1); Y(2*n + 1:end) = -1;    % Labels
oldD = [X Y];

rng(138);
r11 = sqrt(rand(2*n,1));       % Random radii
t11 = [pi/2*rand(n,1); (pi/2*rand(n,1)+pi)];  % Random angles for Q1 and Q3
X11 = [r11.*cos(t11) r11.*sin(t11)];     % Polar-to-Cartesian conversion
r22 = sqrt(rand(2*n,1));
t22 = [pi/2*rand(n,1)+pi/2;
(pi/2*rand(n,1)-pi/2)]; %    Random angles for Q2 and Q4
X22 = [r22.*cos(t22) r22.*sin(t22)];
XX = [X11; X22]; % Predictors
YY = ones(4*n,1); YY(2*n + 1:end) = -1;

predictions = TRAIN0.predictFcn(XX);
confusionmat(YY, predictions)
