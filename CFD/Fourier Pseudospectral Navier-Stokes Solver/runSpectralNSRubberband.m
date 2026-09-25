% runSpectralNSRubberband.m is an initializer function to run standard
% rubberband experiment from IB2d using the method from Chen et al. 

clear
close all 
clc


% domain is [a0,bx]x[a0,by]
a0 = 0; 
bx = 1;
by = 1;

% number of points
Nx=2^7;
Ny=2^7;

% major and minor axis of elipse
b=.4; % for vertically stretched elipse 
a=.2;

% number of lagrangian points
Nlag=Nx*2;

% spring constant
c=2.5e4;

% fluid parameters
rho=1;
mu=0.01;
Re=1;
nu=1/Re;

% timestepping
itermax=1.5; %max time
dt=1e-3;

% run simuluation
filename = SpectralNSRubberband(a0, bx, by, Nx, Ny, b, a, Nlag, c, rho, mu, Re, nu, itermax, dt);

% to only run plotting on a prexisting .h5 file enter name here and comment
% out function call above
% filename='k25000rubberband6_2_21_31.h5';

% plotting
% generates still figures and slice plots at final timestep of speed
% pressure and vorticity
plotrubberbandfigures(filename)

% same as above but adds comparisons with comparison to IB2d
% if .vtk files from IB2d are not available comment out
% IB2dfilename='./olddata/k2.5e4skip20128';
% plotrubberbandfigurescompIB2d(filename, IB2dfilename)

% generates .mp4 movies for speed, vort, pressure
plotrubberbandmovies(filename)


