function plotrubberbandfigures(filename, IB2dfilename)
% generates the following plots: speed, slice plots of speed, vorticity,
% slice plots of vorticity, pressure, slice plots of pressure
[Nx, Ny, Lx, Ly, b, a, Nlag, c, rho, mu, Re, nu, itermax, dt,Lagpointsrecord,Urecord,Vrecord,f1mrecord,f2mrecord]=readoutputfromHDF5(filename);

numsave=75;
IB2dLag = MylagVTKreadskip20(IB2dfilename,numsave, Nlag);
[IB2dU,IB2dV]= MyuvVTKread(IB2dfilename, numsave, Nx);

U=Urecord(:,:,1501);
V=Vrecord(:,:,1501);
Lagpoints=Lagpointsrecord(:,:,1501);
Uexact=IB2dU(:,:,75);
Vexact=IB2dV(:,:,75);

dx=Lx/Nx;
dy=Ly/Ny;

x=[0:Nx-1]*dx;
y=[0:Ny-1]*dy;

[Y,X] = meshgrid(x,y);

kx=[0:Nx/2,-Nx/2+1:-1]*2*pi/Lx;
ky=[0:Ny/2,-Ny/2+1:-1]*2*pi/Ly;

[Ky,Kx]=meshgrid(kx,ky);

K2d=Kx.^2+Ky.^2;

f1hat=fft2(f1mrecord(:,:,1501));
f2hat=fft2(f2mrecord(:,:,1501));
% rho=rho;
% nu=myloader.nu;

% for slcie plots
plot1style='b-';
plot2style='r-.';


%% plot speed
% calculate speed
numericalspeed= getspeed(U,V);
truespeed= getspeed(Uexact,Vexact);
speeddiff=numericalspeed-truespeed;
errorspeed=dx*norm(speeddiff(:),2); % error in 2 norm
figure
pcolor(X,Y,numericalspeed)
hold on 
plot(Lagpoints(1,:),Lagpoints(2,:),'ko')
colorbar
axis equal
% caxis([-1 1])
shading interp
title('speed')
speedtext=['L2 error = ',num2str(errorspeed),', t=1.5'];
title('Standard rubberband speed',speedtext)

% plot slice plot 
quarterrow=floor(Ny*.5);
figure 
plot((1:Nx)/Nx,numericalspeed(quarterrow,:),plot1style)
hold on 
plot((1:Nx)/Nx,truespeed(quarterrow,:),plot2style)
speedtitle = 'Slice Plots of Flow Speed';
speedtitle2 = ['x = ',num2str(quarterrow*dx),', t=1.5'];
title(speedtitle,speedtitle2)
% speedlegend1 = 'Chen et al';
% speedlegend2 = 'IB2d';


%% vorticity plots 
% calculate vorticity
numericalvort = getvorticity(fft2(U),fft2(V),Kx,Ky);
truevort = getvorticity(fft2(Uexact),fft2(Vexact),Kx,Ky);
vortdiff=numericalvort-truevort;
errorvort=dx*norm(vortdiff(:),2) % error in 2 norm
figure
% title('Vorticity');
pcolor(X,Y,numericalvort)
hold on
plot(Lagpoints(1,:),Lagpoints(2,:),'ko')
colorbar
% caxis([-1 1])
shading interp
axis equal
% vorttext=['L2 error = ',num2str(errorvort),', t=1.5'];
title('Standard Rubber Band Vorticity')
skip = 3;
quiver(x(1:skip:end), y(1:skip:end), ...
       U(1:skip:end, 1:skip:end), ...
       V(1:skip:end, 1:skip:end),0.6, 'k');

% slice plots
figure 
plot((1:Nx)/Nx,numericalvort(quarterrow,:),plot1style)
hold on 
plot((1:Nx)/Nx,truevort(quarterrow,:),plot2style)
vorttitle = 'Slice Plots of Flow Vorticity';
speedtitle2 = ['x = ',num2str(quarterrow*dx),', t=1.5'];
title(vorttitle,speedtitle2)

%% plot pressure
% pressure potting and error
Pressnumerical = pressurecalc(fft2(U),fft2(V),Kx,Ky,K2d,f1hat,f2hat,rho,nu,X,Y);
truePress = pressurecalc(fft2(Uexact),fft2(Vexact),Kx,Ky,K2d,f1hat,f2hat,rho,nu,X,Y);
pressdiff=Pressnumerical-truePress;
errorpress=dx*norm(pressdiff(:),2) % error in 2 norm
figure
pcolor(X,Y,Pressnumerical)
hold on 
plot(Lagpoints(1,:),Lagpoints(2,:),'ko')
xlabel('x')
ylabel('y')
axis equal
colorbar
% caxis([-1 1])
shading interp
presstext=['L2 error = ',num2str(errorpress),', t=1.5'];
title('Standard Rubberband Pressure',presstext)

% slice plots
figure 
plot((1:Nx)/Nx,Pressnumerical(quarterrow,:),plot1style)
hold on 
plot((1:Nx)/Nx,truePress(quarterrow,:),plot2style)
presstitle = 'Slice Plots of Flow Pressure';
speedtitle2 = ['x = ',num2str(quarterrow*dx),', t=1.5'];
title(presstitle,speedtitle2)



