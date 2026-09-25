function plotrubberbandmovies(filename)
% plotting for dynamic rubberband skip 20
% generates .mp4 movies of speed vorticity and pressure

numsave=75;
dx=1/128;
[Nx, Ny, Lx, Ly, b, a, Nlag, c, rho, mu, Re, nu, itermax, dt,Lagpointsrecord,Urecord,Vrecord,f1mrecord,f2mrecord]=readoutputfromHDF5(filename);

U=Urecord(:,:,:);
myU=U(:,:,1:20:1501);
V=Vrecord(:,:,:);
myV=V(:,:,1:20:1501);
Lagpoints=Lagpointsrecord(:,:,:);
myLag=Lagpoints(:,:,1:20:1501);

dx=Lx/Nx;
dy=Ly/Ny;

x=[0:Nx-1]*dx;
y=[0:Ny-1]*dy;

[Y,X] = meshgrid(x,y);

kx=[0:Nx/2,-Nx/2+1:-1]*2*pi/Lx;
ky=[0:Ny/2,-Ny/2+1:-1]*2*pi/Ly;

[Ky,Kx]=meshgrid(kx,ky);

K2d=Kx.^2+Ky.^2;

f1hat=fft2(f1mrecord(:,:,:));
myf1m=f1mrecord(:,:,1:20:1501);
f2hat=fft2(f2mrecord(:,:,:));
myf2m=f2mrecord(:,:,1:20:1501);


writerObj = VideoWriter('DynamicBalloonspeedmovie','MPEG-4');
writerObj.FrameRate=30;
open(writerObj);

fig4=figure;
for j=1:numsave
    U=myU(:,:,j);
    V=myV(:,:,j);
    Lagpoints=myLag(:,:,j);

    clf
    % tiledlayout(1,3)
    numericalspeed= getspeed(U,V);
    pcolor(X,Y,numericalspeed)
    hold on
    plot(Lagpoints(1,:),Lagpoints(2,:),'ko')
    colorbar
    caxis([0 1])
    shading interp
    axis equal
    subtitle=['k = ',num2str(c),', mu = ',num2str(mu),', rho = ',num2str(rho)];
    title('Standard Rubberband Speed',subtitle)
    
    F=getframe(fig4);
    writeVideo(writerObj,F)
end
close(writerObj)

writerObj = VideoWriter('DynamicBalloonvortmovie','MPEG-4');
writerObj.FrameRate=30;
open(writerObj);

fig5=figure;
for j=1:numsave
    U=myU(:,:,j);
    V=myV(:,:,j);
    Lagpoints=myLag(:,:,j);

    clf
    
    % calculate vorticity
    numericalvort = getvorticity(fft2(U),fft2(V),Kx,Ky);
    pcolor(X,Y,numericalvort)
    hold on
    plot(Lagpoints(1,:),Lagpoints(2,:),'ko')
    colorbar
    caxis([-30 30])
    shading interp
    axis equal
    subtitle=['k = ',num2str(c),', mu = ',num2str(mu),', rho = ',num2str(rho)];
    title('Standard Rubber Band Vorticity',subtitle)
    skip = 3;
    quiver(x(1:skip:end), y(1:skip:end), ...
           U(1:skip:end, 1:skip:end), ...
           V(1:skip:end, 1:skip:end),0.6, 'k');

    
    F=getframe(fig5);
    writeVideo(writerObj,F)
end
close(writerObj)

writerObj = VideoWriter('DynamicBalloonpressmovie','MPEG-4');
writerObj.FrameRate=30;
open(writerObj);

fig6=figure;
for j=1:numsave
    U=myU(:,:,j);
    V=myV(:,:,j);
    f1m=myf1m(:,:,j);
    f2m=myf2m(:,:,j);
    Lagpoints=myLag(:,:,j);

    clf
    % pressure potting and error
    Pressnumerical = pressurecalc(fft2(U),fft2(V),Kx,Ky,K2d,fft2(f1m),fft2(f2m),rho,nu,X,Y);
    pcolor(X,Y,Pressnumerical)
    hold on
    plot(Lagpoints(1,:),Lagpoints(2,:),'k-')
    xlabel('x')
    ylabel('y')
    colorbar
    caxis([-2 2])
    axis equal
    subtitle=['k = ',num2str(c),', mu = ',num2str(mu),', rho = ',num2str(rho)];
    shading interp
    title('Standard Rubberband Pressure',subtitle)

    
    F=getframe(fig6);
    writeVideo(writerObj,F)
end
close(writerObj)
