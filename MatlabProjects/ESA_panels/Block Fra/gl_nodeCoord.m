function [GN_coord] = gl_nodeCoord(N,nx,lx,ly)
% function [GN_coord] = gl_nodeCoord(N,nx)
% Create a matrix [Nx2] of the global nodes coordinates where:
% 
% input:
% * N  : total number of global nodes
% * nx : number of elements in x direction
% * lx : element x length
% * ly : element y length
%
% output:
% * GN_coord(i,:) : (x,y) coordinates of global node i
%

GN_coord = zeros(N,2);
for i = 1:N
    R = floor((i-1)/(nx+1))+1; % global node i row
    C = mod((i-1),nx+1)+1;     % global node i column
    GN_coord(i,1) = (C-1)*lx;
    GN_coord(i,2) = (R-1)*ly;
end