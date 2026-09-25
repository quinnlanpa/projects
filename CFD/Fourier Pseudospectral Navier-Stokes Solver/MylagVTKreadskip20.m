function lagpoints= MylagVTKreadskip20(fileloc, totaliter, numpoints)

lagpoints=zeros(3,numpoints,totaliter+1); %plus 1 becuase of zero
for k=0:totaliter
    
    % filename=sprintf('./olddata/k2.5e3skip20128/viz_IB2d/lagsPts.%04d.vtk',k);
    
    filename=sprintf('%s/viz_IB2d/lagsPts.%04d.vtk',fileloc,k);
    % data=readVTK(filename)
    % keyboard
    fid = fopen(filename, 'r');
    
    % Skip header lines until we hit POINTS
    while true
        line = fgetl(fid);
        if contains(line, 'POINTS')
            % Extract number of points from this line e.g. "POINTS 64 float"
            parts = strsplit(line);
            N = str2double(parts{2});
            break
        end
    end
    
    % Read the point data (N rows of x, y, z)
    data = fscanf(fid, '%f', [3, N])';  % Nx3 matrix
    fclose(fid);
    
    x = data(:, 1);
    y = data(:, 2);
    z = data(:, 3);

    lagpoints(:,:,k+1)=[x';y';z'];
end
% keyboard