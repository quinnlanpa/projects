function filename=writeoutputtoHDF5(Nx, Ny, Lx, Ly, b, a, Nlag, c, rho, mu, Re, nu, itermax, dt,Lagpointsrecord,Urecord,Vrecord,f1mrecord,f2mrecord)

filename=['k',num2str(c),'rubberband',char(datetime('now','Format','M_d_H_m')),'.h5'];

h5create(filename,'/Ny',size(Ny))
h5create(filename,'/Nx',size(Nx))
h5write(filename,'/Ny',Ny)
h5write(filename,'/Nx',Nx)

h5create(filename,'/Ly',size(Ly))
h5create(filename,'/Lx',size(Lx))
h5write(filename,'/Ly',Ly)
h5write(filename,'/Lx',Lx)

h5create(filename,'/a',size(a))
h5create(filename,'/b',size(b))
h5write(filename,'/a',a)
h5write(filename,'/b',b)

h5create(filename,'/Nlag',size(Nlag))
h5create(filename,'/c',size(c))
h5write(filename,'/Nlag',Nlag)
h5write(filename,'/c',c)

h5create(filename,'/rho',size(rho))
h5create(filename,'/mu',size(mu))
h5create(filename,'/Re',size(Re))
h5create(filename,'/nu',size(nu))
h5write(filename,'/rho',rho)
h5write(filename,'/mu',mu)
h5write(filename,'/Re',Re)
h5write(filename,'/nu',nu)

h5create(filename,'/itermax',size(itermax))
h5create(filename,'/dt',size(dt))
h5write(filename,'/itermax',itermax)
h5write(filename,'/dt',dt)

h5create(filename,'/Lagpointsrecord',size(Lagpointsrecord))
h5create(filename,'/Urecord',size(Urecord))
h5create(filename,'/Vrecord',size(Vrecord))
h5create(filename,'/f1mrecord',size(f1mrecord))
h5create(filename,'/f2mrecord',size(f2mrecord))

h5write(filename,'/Lagpointsrecord',Lagpointsrecord)
h5write(filename,'/Urecord',Urecord)
h5write(filename,'/Vrecord',Vrecord)
h5write(filename,'/f1mrecord',f1mrecord)
h5write(filename,'/f2mrecord',f2mrecord)

