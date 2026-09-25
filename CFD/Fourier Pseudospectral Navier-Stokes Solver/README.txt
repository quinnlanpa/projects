README SpectralNSRubberband
Implements Navier-Stokes solver from Chen et al. 

Must be ran through runSpectralNSRubberband.m which initializes parameters. At the end of running the parameters and fluid properties are saved into an .h5 file. A new .h5 file is made every time the simulation runs. This file name will have the format k25000rubberband6_2_14_3.h5. K25000 represents the spring constant parameters k=2.5e4. 6_2_14_3 represents that the simulation was ran on 6/2 at 2:03pm. This was done this way because if the date and time were not included in the name the h5create() function may cause issues. 

Plotting is handled in separate functions. These can be ran directly after the simulation or the simulation can be commented out and the plotting can happen from a user entered .h5 file. 

plotting functions. 
plotrubberbandfigures() generates pcolor plots and slice plots of speed vorticity and pressure

plotrubberbandfigurescompIB2d() is the same as above however it reads in .vtk files from IB2d and give comparisons in slice plots and errors on the pcolor plots. 

plotrubberbandmovies saves a .mp4 movie for speed, vorticity and pressure through all timesteps of the simulation. 

all of these functions can be ran independently of each other and only need passed associated filenames.   