function [U,V]= MyuvVTKread(fileloc, totaliter, numpoints)
% Read all VTK files u_0000.vtk to u_0075.vtk

nFiles = totaliter;  % 0000 to 0075

% Grid is 32x32 from the header
nx = numpoints;
ny = numpoints;
nPoints = nx * ny;

% Preallocate storage: ux, uy each (nx, ny, nFiles)
U = zeros(nx, ny, nFiles);
V = zeros(nx, ny, nFiles);
time_all = zeros(1, nFiles);

for k = 1:nFiles
    % filename = sprintf('./viz_IB2d/u.%04d.vtk', k-1);  % u_0000.vtk, u_0001.vtk, ...
    filename=sprintf('%s/viz_IB2d/u.%04d.vtk',fileloc,k);

    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end

    % --- Parse header ---
    fgetl(fid);  % # vtk DataFile Version 2.0
    fgetl(fid);  % Comment
    fgetl(fid);  % ASCII
    fgetl(fid);  % blank line
    fgetl(fid);  % DATASET STRUCTURED_POINTS
    fgetl(fid);  % FIELD FieldData 1
    fgetl(fid);  % TIME 1 1 double

    % Read time value
    time_all(k) = fscanf(fid, '%f', 1);

    fgetl(fid);  % finish the time line
    fgetl(fid);  % DIMENSIONS ...
    fgetl(fid);  % blank line
    fgetl(fid);  % ORIGIN ...
    fgetl(fid);  % SPACING ...
    fgetl(fid);  % blank line
    fgetl(fid);  % POINT_DATA ...
    fgetl(fid);  % VECTORS u double
    fgetl(fid);  % blank line

    % --- Read vector data ---
    % Each point has 3 values: ux uy uz
    data = fscanf(fid, '%f', [3, nPoints]);  % reads as 3 x nPoints

    fclose(fid);

    % data(1,:) = ux, data(2,:) = uy, data(3,:) = uz
    % Reshape from column-major order into (nx, ny)
    U(:,:,k) = reshape(data(1,:), nx, ny);
    V(:,:,k) = reshape(data(2,:), nx, ny);
end
