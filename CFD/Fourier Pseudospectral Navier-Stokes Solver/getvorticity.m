function vort = getvorticity(Uhat,Vhat,Kx,Ky)
% this function finds the vorticity of a velocity field
% from the true function
% speed is calculated as curl([u;v])
vort=real(ifft2(1i*Kx.*Vhat-1i*Ky.*Uhat));
