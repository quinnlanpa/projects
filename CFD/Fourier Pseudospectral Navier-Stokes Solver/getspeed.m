function speed = getspeed(U,V)
% this function finds the speed of a velocity field and returns its error
% from the true function
% speed is calculated as sqrt(u^2+v^2)
speed=sqrt(U.^2+V.^2);
