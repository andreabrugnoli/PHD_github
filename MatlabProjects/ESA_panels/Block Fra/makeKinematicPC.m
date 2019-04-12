function[tauCP] = makeKinematicPC(GN_coord,P_id,C_ids)

Pxy = GN_coord(P_id,:); % Node P coordinates
tauCP = zeros(3,3,length(C_ids));
for i = 1: length(C_ids)
    dx = GN_coord(C_ids(i),1) - Pxy(1); % x-distance of node C(i) from node P
    dy = GN_coord(C_ids(i),2) - Pxy(2); % y-distance of node C(i) from node P
    tauCP(:,:,i) =[1 dy -dx;0 1 0;0 0 1];
end

end