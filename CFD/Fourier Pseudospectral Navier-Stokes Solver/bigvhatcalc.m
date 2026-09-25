function [bigvhatx,bigvhaty] = bigvhatcalc(Uhat,Vhat,Kx,Ky,K2d,f1hat,f2hat,rho,nu,X,Y)
% Calculates bigvhat from chen et al. paper 

% for finding nonlinear term
U=real(ifft2(Uhat));
V=real(ifft2(Vhat));
% find nonlinear term shat
dudx=real(ifft2(1i*Kx.*Uhat));
dudy=real(ifft2(1i*Ky.*Uhat));
dvdx=real(ifft2(1i*Kx.*Vhat));
dvdy=real(ifft2(1i*Ky.*Vhat));


% handle aliasing
% kmax=floor(size(Uhat,1)*3/2);
kmax=floor(size(Uhat,1)*1/3);
orszag23 = (abs(Kx) <=kmax) & (abs(Ky)<=kmax);

% nonlinear term must be computed in physical space
shatx=fft2(U.*dudx+V.*dudy).*orszag23;
shaty=fft2(U.*dvdx+V.*dvdy).*orszag23;


Nx=size(K2d,1);
Ny=size(K2d,2);
Nzero=1; %my ordering 
K2d(Nzero,Nzero)=1;


% new approach solve laplacian(p)=div(f/rho-s)
rhshat=1i*Kx.*(f1hat/rho-shatx)+1i*Ky.*(f2hat/rho-shaty);

phat=-rhshat./(K2d); 
% set mean
phat(1,1)=0*Nx*Ny;

% find bigvhat=(f1hat/rho-shatx)-grad(phat)
bigvhatx3=(f1hat/rho-shatx)-1i*Kx.*phat;
bigvhaty3=(f2hat/rho-shaty)-1i*Ky.*phat;

bigvhatx=bigvhatx3;
bigvhaty=bigvhaty3;

