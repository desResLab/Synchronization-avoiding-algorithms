clc
clear
close all
cases = 100;
steady = -1;
x = linspace(0,25,100);

for c = 1:cases
    max_dz = 0 + (steady)*rand;
    
    y = @(x) max_dz/25/25*x.^2 ;
    plot(x,y(x))
    hold  on
    
end