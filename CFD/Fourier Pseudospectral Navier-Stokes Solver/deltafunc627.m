function phi = deltafunc627(dist)
% delta function eqn 6.27 in PeskinActa2002
if dist <= -2
    phi=0;
elseif dist <= -1
    phi=(1/8)*(5+2*dist-sqrt(-7-12*dist-4*dist^2));
elseif dist <=0
    phi=(1/8)*(3+2*dist+sqrt(1-4*dist-4*dist^2));
elseif dist <=1
    phi=(1/8)*(3-2*dist+sqrt(1+4*dist-4*dist^2));
elseif dist <=2
    phi=(1/8)*(5-2*dist-sqrt(-7+12*dist-4*dist^2));
elseif dist > 2
    phi=0;
else 
    % error dist not caught in if statment 
    keyboard
end

% delta function eqn 6.27 in PeskinActa2002
% if dist < -2
%     phi=0;
% elseif dist <= -1
%     phi=(1/8)*(5-2*abs(dist)-sqrt(-7+12*abs(dist)-4*dist^2));
% elseif dist <=0
%     phi=(1/8)*(3-2*abs(dist)+sqrt(1+4*abs(dist)-4*dist^2));
% elseif dist <=1
%     phi=(1/8)*(3-2*abs(dist)+sqrt(1+4*abs(dist)-4*dist^2));
% elseif dist <=2
%     phi=(1/8)*(5-2*abs(dist)-sqrt(-7+12*abs(dist)-4*dist^2));
% elseif dist > 2
%     phi=0;
% else 
%     % error dist not caught in if statment 
%     keyboard
% end