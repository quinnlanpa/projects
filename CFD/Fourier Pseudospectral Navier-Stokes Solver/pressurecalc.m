function P = pressurecalc(Uhat,Vhat,Kx,Ky,K2d,f1hat,f2hat,rho,nu,X,Y)
% for calculating pressure explicitly.
% code is lifted from bigvhatcalc3.m

% for finding nonlinear term
U=real(ifft2(Uhat));
V=real(ifft2(Vhat));
% find nonlinear term shat
dudx=real(ifft2(1i*Kx.*Uhat));
dudy=real(ifft2(1i*Ky.*Uhat));
dvdx=real(ifft2(1i*Kx.*Vhat));
dvdy=real(ifft2(1i*Ky.*Vhat));

% 0 to run normally 1 for accuracy and plots ------------------------------
testingmode=0;
% -------------------------------------------------------------------------

% handle aliasing
kmax=floor(size(Uhat,1)*1/3);
orszag23 = (abs(Kx) <=kmax) & (abs(Ky)<=kmax);
% keyboard

% nonlinear term must be computed in physical space
shatx=fft2(U.*dudx+V.*dudy).*orszag23;
shaty=fft2(U.*dvdx+V.*dvdy).*orszag23;

% shatx=fft2(U.*dudx+V.*dudy);
% shaty=fft2(U.*dvdx+V.*dvdy);
% shatx(kmax+1:end-kmax,:)=0;
% shatx(:,kmax+1:end-kmax)=0;
% shaty(kmax+1:end-kmax,:)=0;
% shaty(:,kmax+1:end-kmax)=0;
% [shatx,shaty]= nonlinear_3over2(Uhat, Vhat);

Nx=size(K2d,1);
Ny=size(K2d,2);
Nzero=1; %my ordering 
K2d(Nzero,Nzero)=1;

% keyboard

% new approach solve laplacian(p)=div(f/rho-s)
rhshat=1i*Kx.*(f1hat/rho-shatx)+1i*Ky.*(f2hat/rho-shaty);
phat=-rhshat./(K2d); 
% set mean
phat(1,1)=0*Nx*Ny;
P=real(ifft2(phat));
