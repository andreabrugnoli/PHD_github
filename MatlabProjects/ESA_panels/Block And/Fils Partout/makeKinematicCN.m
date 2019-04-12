function[tauCN] = makeKinematicCN(GN_coord,C_ids,Cs_xy)

[nnodes, ~] = size(Cs_xy);

tauCN = zeros(3,3,nnodes);
for i = 1: nnodes
    dx = GN_coord(C_ids(i),1) - Cs_xy(i,1); % x-distance of node C(i) from its closest node
    dy = GN_coord(C_ids(i),2) - Cs_xy(i,2); % y-distance of node C(i) from its closest node
    tauCN(:,:,i) =[1 dy -dx;0 1 0;0 0 1];
end

end