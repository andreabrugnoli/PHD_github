function[K2,M2] = makeClampP(GN_coord,Kas,Mas,P_id)
% function[K2,M2] = makeClampP(GN_coord,Kas,Mas,P_id)
%
% Input:
% * GN_coord : Global nodes coordinates
% * Kas      : Assembled Kinetic Matrix
% * Mas      : Assembled Mass Matrix
% * P_id     : Clamped Global Node ID
%
% Output:
% * K2 : Constrained Kinetic Matrix
% * M2 : Constrained Mass Marix
%

Pxy = GN_coord(P_id,:);
N = length(GN_coord(:,1)); % Total number of global nodes

P = zeros(3*N,3);
for i = 1:N
    dx = GN_coord(i,1) - Pxy(1); % x-distance of node i from node P
    dy = GN_coord(i,2) - Pxy(2); % y-distance of node i from node P
    P((3*i-2):3*i,:) = [1 dy -dx;0 1 0;0 0 1];
end


P = [[eye(3*(P_id-1),3*(P_id-1));zeros(3*(N-P_id+1),3*(P_id-1))] P [zeros(3*P_id,3*(N-P_id));eye(3*(N-P_id),3*(N-P_id))]];


K2 = P'*Kas*P;
M2 = P'*Mas*P;

end
