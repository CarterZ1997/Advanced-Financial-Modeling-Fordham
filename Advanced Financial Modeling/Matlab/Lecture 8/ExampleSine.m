
rng(38);
x = -4*pi:0.01:4*pi;
y=sin(x);
x=x';
y=y';
r=0.8*(rand(size(x))-0.5);
y=y+r;
z=[x y];
plot(x,y), grid on

rng(40);
x1 = -4*pi:0.01:4*pi;
y1 = sin(x1);
x1 = x1';
y1 = y1';
y1 = y1 + r;
z1 = [x1 y1];
