function [Nx, Ny, Lx, Ly, b, a, Nlag, c, rho, mu, Re, nu, itermax, dt,Lagpointsrecord,Urecord,Vrecord,f1mrecord,f2mrecord]=readoutputfromHDF5(filename)

% filename=['k',num2str(c),'rubberband',char(datetime('now','Format','M_d_H_m'))];

Ny=h5read(filename,'/Ny');
Nx=h5read(filename,'/Nx');

Ly=h5read(filename,'/Ly');
Lx=h5read(filename,'/Lx');

a=h5read(filename,'/a');
b=h5read(filename,'/b');

Nlag=h5read(filename,'/Nlag');
c=h5read(filename,'/c');

rho=h5read(filename,'/rho');
mu=h5read(filename,'/mu');
Re=h5read(filename,'/Re');
nu=h5read(filename,'/nu');

itermax=h5read(filename,'/itermax');
dt=h5read(filename,'/dt');

Lagpointsrecord=h5read(filename,'/Lagpointsrecord');
Urecord=h5read(filename,'/Urecord');
Vrecord=h5read(filename,'/Vrecord');
f1mrecord=h5read(filename,'/f1mrecord');
f2mrecord=h5read(filename,'/f2mrecord');

