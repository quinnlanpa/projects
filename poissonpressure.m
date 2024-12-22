% Quinn Aiken
% MATH 447 Computational Fluids
% Solves the pressure poisson formuation of the Stokes Equations on domain
% [a,b] x [a,b]. The formulation is two equations laplace(p)=div(f), 
% and laplace([u,v])=grad(p)+f using finite differences, periodic
% boundary conditions and uniform grid spacing.



clc
clear all
close all
% endpoints of domain
a = 0; 
b = 1; 
m = 3000; % resolution of grid points


h = (b-a)/(m+1);
x = linspace(a,b-h,m+1);   % grid points x including boundaries
y = linspace(a,b-h,m+1);   % grid points y including boundaries


% make a uniform grid for evaluating over
[X,Y] = meshgrid(x,y);      
X = X';                     
Y = Y';                     


% given exterior forces
f1 = @(x,y) 8*pi^2*sin(2*pi*x).*sin(2*pi*y)-2*pi*sin(2*pi*x); 
f2 = @(x,y) 8*pi^2*cos(2*pi*x).*cos(2*pi*y)-2*pi*sin(2*pi*y);

% true solution for test problem
ptrue = cos(2*pi*X)+cos(2*pi*Y);
utrue = sin(2*pi*X).*sin(2*pi*Y);
vtrue = cos(2*pi*X).*cos(2*pi*Y);


% exterior forces over grid points
f1m = f1(X,Y);         
f2m = f2(X,Y);        

% initialize rhs of system for solving for p using finite differences
% according to laplace(p)=div(f)
rhs1=zeros(m+1,m+1);
%interior of matrix
for i=2:m
    for j=2:m
        rhs1(i,j)=(1/(2*h))*(f1m(i+1,j)-f1m(i-1,j))+(1/(2*h))*(f2m(i,j+1)-f2m(i,j-1));
    end
end

%i=0 j from 2 to m and i=m+1
for j=2:m
    rhs1(1,j)=(1/(2*h))*(f1m(2,j)-f1m(m+1,j))+(1/(2*h))*(f2m(1,j+1)-f2m(1,j-1));
    rhs1(m+1,j)=(1/(2*h))*(f1m(1,j)-f1m(m,j))+(1/(2*h))*(f2m(m+1,j+1)-f2m(m+1,j-1));
end

%j=0 i from 2 to m and j=m+1
for i=2:m
    rhs1(i,1)=(1/(2*h))*(f1m(i+1,1)-f1m(i-1,1))+(1/(2*h))*(f2m(i,2)-f2m(i,m+1));
    rhs1(i,m+1)=(1/(2*h))*(f1m(i+1,m+1)-f1m(i-1,m+1))+(1/(2*h))*(f2m(i,1)-f2m(i,m));
end

%four corners of domain
%i=0 j from 2 to m and i=m+1
rhs1(1,1)=(1/(2*h))*(f1m(2,1)-f1m(m+1,1))+(1/(2*h))*(f2m(1,2)-f2m(1,m+1));
rhs1(1,m+1)=(1/(2*h))*(f1m(2,m+1)-f1m(m+1,m+1))+(1/(2*h))*(f2m(1,1)-f2m(1,m));
rhs1(m+1,1)=(1/(2*h))*(f1m(1,1)-f1m(m,1))+(1/(2*h))*(f2m(m+1,2)-f2m(m+1,m+1));
rhs1(m+1,m+1)=(1/(2*h))*(f1m(1,m+1)-f1m(m,m+1))+(1/(2*h))*(f2m(m+1,1)-f2m(m+1,m));


% convert the 2d grid function rhs into a column vector for rhs of system:
F = reshape(rhs1,(m+1)*(m+1),1);

% form matrix A:
I = speye(m+1);
e = ones(m+1,1);
T = spdiags([e -2*e e],[-1 0 1],m+1,m+1);
S = spdiags([e e],[-1 1],m+1,m+1);
T(1,m+1)=1;
T(m+1,1)=1;
A = (kron(I,T)+kron(T,I)) / (h^2);


% Solve the linear system for p and vecterize results
pvec = A\F;  
psoln = reshape(pvec,m+1,m+1);

% generate new rhs for finding u and v using finite difference
% acording to laplace([u,v])=grad(p)+f
rhs2a=zeros(m+1,m+1);
rhs2b=zeros(m+1,m+1);

% for u 
for j=1:m+1
    rhs2a(1,j)=(1/(2*h))*(psoln(2,j)-psoln(m+1,j))-f1m(1,j);
    rhs2a(m+1,j)=(1/(2*h))*(psoln(1,j)-psoln(m,j))-f1m(m+1,j);
end
for i=2:m
    for j=1:m+1
        rhs2a(i,j)=(1/(2*h))*(psoln(i+1,j)-psoln(i-1,j))-f1m(i,j);
    end
end

% for v
for i=1:m+1
   rhs2b(i,1)=(1/(2*h))*(psoln(i,2)-psoln(i,m+1))-f2m(i,1); % j=1
   rhs2b(i,m+1)=(1/(2*h))*(psoln(i,1)-psoln(i,m))-f2m(i,m+1); % j=m+1
end
for j=2:m
    for  i=1:m+1
        rhs2b(i,j)=(1/(2*h))*(psoln(i,j+1)-psoln(i,j-1))-f2m(i,j);
    end
end

% compute the rhs of system for u&v
F2a = reshape(rhs2a,(m+1)*(m+1),1);
F2b = reshape(rhs2b,(m+1)*(m+1),1);

% solve for u and vecterize 
uvec = A\F2a;  
usoln = reshape(uvec,m+1,m+1);

% solve for v and vecterize
vvec = A\F2b;  
vsoln = reshape(vvec,m+1,m+1);

% assuming true solution is known and stored in utrue:
err = max(max(abs(psoln-ptrue)));   
fprintf('Error relative to true solution of pressure = %10.3e \n',err)

erru = max(max(abs(usoln-utrue)));   
fprintf('Error relative to true solution of u = %10.3e \n',erru)

errv = max(max(abs(vsoln-vtrue)));   
fprintf('Error relative to true solution of v = %10.3e \n',errv)



clf
hold on

% surf plots of u v and p:

tiledlayout(3,2)
nexttile

surf(X,Y,ptrue,'EdgeColor','none');
axis([0 1 0 1 -3 3])
shading interp
title('True Solution of Pressure')
xlabel('x')
ylabel('y')
zlabel('z')
nexttile

surf(X,Y,psoln,'EdgeColor','none');
axis([0 1 0 1 -3 3])
shading interp
title('Computed Solution of Pressure')
xlabel('x')
ylabel('y')
zlabel('z')
nexttile


surf(X,Y,utrue,'EdgeColor','none');
axis([0 1 0 1 -2 2])
shading interp
title('True solution of u')
xlabel('x')
ylabel('y')
zlabel('z')
nexttile

surf(X,Y,usoln,'EdgeColor','none');
axis([0 1 0 1 -2 2])
shading interp
title('Computed Solution of u')
xlabel('x')
ylabel('y')
zlabel('z')
nexttile

surf(X,Y,vsoln,'EdgeColor','none');       
shading interp
title('Computed Solution of v')
xlabel('x')
ylabel('y')
zlabel('z')
nexttile

surf(X,Y,vtrue,'EdgeColor','none');       
shading interp
title('True Solution of v')
xlabel('x')
ylabel('y')
zlabel('z')

t.Padding = 'compact';
t.TileSpacing = 'compact';

figure

% contour plot figure
tiledlayout(3,2)
nexttile

contour(X,Y,psoln,30,'k')
title('Contour plot of computed solution of pressure')
nexttile

contour(X,Y,ptrue,30,'k')
title('Contour plot of true solution of pressure')
nexttile

contour(X,Y,usoln,30,'k')
title('Contour plot of computed solution of u')
nexttile

contour(X,Y,utrue,30,'k')
title('Contour plot of true solution of u')
nexttile

contour(X,Y,vsoln,30,'k')
title('Contour plot of computed solution of v')
nexttile

contour(X,Y,vtrue,30,'k')
title('Contour plot of true solution of v')



 