function [GN_coord] = gl_nodeCoord(N,nx,lx)
% function [GN_coord] = gl_nodeCoord(N,nx)
% Create a matrix [Nx2] of the global nodes coordinates where:
% 
% input:
% * N  : total number of global nodes
% * nx : number of elements in x direction
% * lx : element x length
%
% output:
% * GN_coord(i,:) : (x,y) coordinates of global node i
%

GN_coord = [];
for i = 1:N
    GN_coord(i) = (i-1)*lx/nx;
end


end