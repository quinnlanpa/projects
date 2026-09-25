function filename = SpectralNSRubberband(a0, bx, by, Nx, Ny, b, a, Nlag, c, rho, mu, Re, nu, itermax, dt)
% IB method as defined by chen pesking paper dynamic balloon/ standard
% rubberband. Full Navier Stokes Solver on periodic boundary conditions.


numsave=75;
%% setup domain and eularien grid 

Lx=bx-a0;
Ly=by-a0;
% number of points

dx=Lx/Nx;
dy=Ly/Ny;

x=[0:Nx-1]*dx;
y=[0:Ny-1]*dy;
% keyboard
% make a uniform eularian grid for evaluating over
% [X,Y] = meshgrid(x,y);
[Y,X] = meshgrid(x,y);


%% setup lagrangian boundary
% now form lagrangian points
% form circle in middle of domain
centerx=Lx/2;
centery=Ly/2;

r=sqrt(a*b);

% find dtheta
dtheta=(0:Nlag-1)*(2*pi/Nlag);
ds=min( Lx/(Nlag), Ly/(Nlag));
% min( Lx/(2*Nx), Ly/(2*Ny)  ds from IB2d line 203 IBMdriver.m

lagpoints=zeros(2,size(dtheta,2)); % x in first row y in second row

lagpoints(1,:)=a*cos(dtheta);
lagpoints(2,:)=b*sin(dtheta);
lagpoints=lagpoints+[centerx; centery];

% setup second derivative matrix
e = ones(size(lagpoints,2),1);
A2 = spdiags([e -2*e e],[-1 0 1],size(lagpoints,2),size(lagpoints,2));
A2(1,end)=1;
A2(end,1)=1;


%% setup wave numbers with scaling because Lx~=2*pi
kx=[0:Nx/2,-Nx/2+1:-1]*2*pi/Lx;
ky=[0:Ny/2,-Ny/2+1:-1]*2*pi/Ly;

[Kx,Ky]=meshgrid(kx,ky);

K2d=Kx.^2+Ky.^2;
K2dNonzero=K2d;
K2dNonzero(1,1)=1;


%% timestepping and data initialization
% initalize u,v,f1,f2 all to zero
U=zeros(size(X));
V=zeros(size(X));
f1m=zeros(size(X));
f2m=zeros(size(X));
errorstore=zeros(1,itermax/dt+1);
lagu=zeros(1,size(lagpoints,2));
lagv=zeros(1,size(lagpoints,2));

%% iteration timing and saving and benchmarking setup
tic
iter=0;
iterationcounter=0;

% saving my lagpoints at each iteration
Lagpointsrecord=zeros(2,size(lagpoints,2),numsave+1);
Lagpointsrecord(:,:,1)=lagpoints;

% saving u and v
Urecord=zeros(size(U,1),size(U,2),numsave+1);
Vrecord=zeros(size(V,1),size(V,2),numsave+1);

% saving f1m and f2m
f1mrecord=zeros(size(f1m,1),size(f1m,2),numsave+1);
f2mrecord=zeros(size(f2m,1),size(f2m,2),numsave+1);

writerObj = VideoWriter('CompDynamicBalloon','MPEG-4');
writerObj.FrameRate=30;
open(writerObj);
fig1=figure(1);

plot(lagpoints(1,:),lagpoints(2,:),'or')
hold on
% plot(IB2dLag(1,:,1),IB2dLag(2,:,1),'*b')
% legend('Navier-Stokes','IB2d')
axis equal
axis([0 Lx 0 Ly])
title('Comparison Dynamic Balloon',['t=',num2str(iter)])
xlabel('X')
ylabel('Y')

while iter<itermax
    %% iteration: take step
    iter=iter+dt;
    iterationcounter=iterationcounter+1;

    %% half step
    lagpointshalf=lagpoints+(dt/2).*[lagu;lagv];

    % save current lagpoints in old to save having to reqrite a bunch
    lagpointsold=lagpoints;
    lagpoints=lagpointshalf;
    
    %% force spreading
    % compute force of current lagragian boundary (F for lagrangian)
    k=c*ds*ds; % necissary for second derivative to be same as IB2d
    lagFx=(k/(ds^2))*A2*lagpoints(1,:)';
    lagFy=(k/(ds^2))*A2*lagpoints(2,:)';    

    % forces based off displacement like IB2d
    % lagFx=zeros(1,size(lagpoints,2));
    % lagFy=zeros(1,size(lagpoints,2));
    % for i=1:size(lagpoints,2)-1
    %     xforce=c*(lagpoints(1,i+1)-lagpoints(1,i));
    %     lagFx(i)=lagFx(i)+xforce;
    %     lagFx(i+1)=lagFx(i+1)-xforce;
    % 
    %     yforce=c*(lagpoints(2,i+1)-lagpoints(2,i));
    %     lagFy(i)=lagFy(i)+yforce;
    %     lagFy(i+1)=lagFy(i+1)-yforce;
    % end
    % % handle periodic forces
    % xforce=c*(lagpoints(1,1)-lagpoints(1,end));
    % lagFx(end)=lagFx(end)+xforce;
    % lagFx(1)=lagFx(1)-xforce;
    % 
    % yforce=c*(lagpoints(2,1)-lagpoints(2,end));
    % lagFy(end)=lagFy(end)+yforce;
    % lagFy(1)=lagFy(1)-yforce;
    % 
    % % make column so consistent
    % lagFx=lagFx';
    % lagFy=lagFy';

    f1m=zeros(size(X));
    f2m=zeros(size(X));
    % lag to eul force spreading
    for i=1:size(lagpoints,2)
        indexj=floor(lagpoints(1,i)/dx+1);
        indexk=floor(lagpoints(2,i)/dy+1);
        for jj=indexj-1:indexj+2
            for kk=indexk-1:indexk+2
                % Periodic wrapping
                j = mod(jj-1, Nx) + 1;
                k = mod(kk-1, Ny) + 1;
                
                rx=(x(j)-lagpoints(1,i))/dx;
                ry=(y(k)-lagpoints(2,i))/dy;
                phix=deltafunc627(rx);
                phiy=deltafunc627(ry);
            
                phitotal=(1/(dx*dy))*phix*phiy;
    
                f1m(k,j)=f1m(k,j)+lagFx(i)*phitotal*ds;
                f2m(k,j)=f2m(k,j)+lagFy(i)*phitotal*ds;
            end
        end
    end
    

    %% fluid solver
    % move into spectral space
    Uhat=fft2(U);
    Vhat=fft2(V);
    f1hat=fft2(f1m);
    f2hat=fft2(f2m);

    % compute bigvhat for BE step
    [bigvhatx,bigvhaty]=bigvhatcalc(Uhat,Vhat,Kx,Ky,K2d,f1hat,f2hat,rho,nu,X,Y);
    % do backward euler for half step
    BE=1+dt*mu*K2d/(2*rho); %coef of lhs

    % compute half timestep
    Uhathalf=(Uhat+(dt/2)*bigvhatx)./BE;
    Vhathalf=(Vhat+(dt/2)*bigvhaty)./BE;

    % recompute bigvhat for CN step
    [bigvhatx2,bigvhaty2]=bigvhatcalc(Uhathalf,Vhathalf,Kx,Ky,K2d,f1hat,f2hat,rho,nu,X,Y);
    CN=1+(mu*dt*K2d)/(2*rho); %coef of lefthand side of eqn

    % A1=(1-(dt*mu*K2d)/(2*rho))./CN;
    % B1=dt./CN;
    % compute next full timestep.
    Uhatmfull=(dt*bigvhatx2+(1-(dt*mu*K2d)/(2*rho)).*Uhat)./CN;
    Vhatmfull=(dt*bigvhaty2+(1-(dt*mu*K2d)/(2*rho)).*Vhat)./CN;
    
    % update for next iteration;
    Uhat=Uhatmfull;
    Vhat=Vhatmfull;
    
    % back to physical space
    U=real(ifft2(Uhat));
    V=real(ifft2(Vhat));
    Uhalf=real(ifft2(Uhathalf));
    Vhalf=real(ifft2(Vhathalf));
    
    %% Velocity interp twice and updating
    lagu=zeros(1,size(lagpoints,2));
    lagv=zeros(1,size(lagpoints,2));
    laguhalf=zeros(1,size(lagpoints,2));
    lagvhalf=zeros(1,size(lagpoints,2));
    % updated for less iteration
    % Velocity interpolation for full step
    for i=1:size(lagpoints,2)
        indexj=floor(lagpoints(1,i)/dx+1);
        indexk=floor(lagpoints(2,i)/dy+1);
        for jj=indexj-2:indexj+2
                for kk=indexk-2:indexk+2
                    % Periodic wrapping
                    % j = mod(jj-1, Nx) + 1;
                    % k = mod(kk-1, Ny) + 1;
                    j=jj;
                    k=kk;

                    rx=(x(j)-lagpoints(1,i))/dx;
                    ry=(y(k)-lagpoints(2,i))/dy;
                    
                    phix=deltafunc627(rx);
                    phiy=deltafunc627(ry);
                    
                    phitotal=(1/(dx*dy))*phix*phiy; % this one is correct
                    % phitotal=phix*phiy;
                    lagu(i)=lagu(i)+U(k,j)*phitotal*(dx^2);
                    lagv(i)=lagv(i)+V(k,j)*phitotal*(dy^2);
                end
        end
        
    end
    
    % Velocity interpolation for half step
    for i=1:size(lagpoints,2)
        indexj=floor(lagpoints(1,i)/dx+1);
        indexk=floor(lagpoints(2,i)/dy+1);
        for jj=indexj-1:indexj+2
                for kk=indexk-1:indexk+2
                    % Periodic wrapping
                    % j = mod(jj-1, Nx) + 1;
                    % k = mod(kk-1, Ny) + 1;
                    j=jj;
                    k=kk;

                    rx=(x(j)-lagpoints(1,i))/dx;
                    ry=(y(k)-lagpoints(2,i))/dy;
                    
                    phix=deltafunc627(rx);
                    phiy=deltafunc627(ry);


                    phitotal=(1/(dx*dy))*phix*phiy; % this one is correct
                    % phitotal=phix*phiy;
                    laguhalf(i)=laguhalf(i)+Uhalf(k,j)*phitotal*(dx*dy);
                    lagvhalf(i)=lagvhalf(i)+Vhalf(k,j)*phitotal*(dx*dy);
                end
        end
        
    end
    
    % updating position of lag points using half step and old points
    lagpoints=lagpointsold+dt.*[laguhalf;lagvhalf];
    %% recording simulation
    Lagpointsrecord(:,:,iterationcounter+1)=lagpoints;
    Urecord(:,:,iterationcounter+1)=Uhalf;
    Vrecord(:,:,iterationcounter+1)=Vhalf;
    f1mrecord(:,:,iterationcounter+1)=f1m;
    f2mrecord(:,:,iterationcounter+1)=f2m;

    %% Plotting
    if mod(iterationcounter,1)==0
        % IB2dcounter=iterationcounter/frameskip;
        % IB2dcounter=IB2dcounter+1;

        % --------------------------------------------------------movie
        
        figure(fig1)
        clf
        plot( lagpoints(1,:),lagpoints(2,:),'or')
        hold on
        % plot(IB2dLag(1,:,iterationcounter+1),IB2dLag(2,:,iterationcounter+1),'*b')
        % legend('Navier-Stokes','IB2d')
        axis equal
        axis([0 Lx 0 Ly])
        title('Comparison Dynamic Balloon',['t=',num2str(iter)])
        xlabel('X')
        ylabel('Y')
        % pause(.001)
    end
end
toc
%% saving output
filename=writeoutputtoHDF5(Nx, Ny, Lx, Ly, b, a, Nlag, c, rho, mu, Re, nu, itermax, dt,Lagpointsrecord,Urecord,Vrecord,f1mrecord,f2mrecord);
