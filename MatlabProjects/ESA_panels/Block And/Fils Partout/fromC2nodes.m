function [C_ids] = fromC2nodes(Cs_xy, GN_coord, P_id)

[nnodes, ~] = size(Cs_xy);
C_ids = zeros(1, nnodes);

for i=1:nnodes
    dist_C_i = (GN_coord(:,1) - Cs_xy(i,1)).^2 + (GN_coord(:,2) - Cs_xy(i,2)).^2;
    
    [~, ind] = min(dist_C_i);
    
    if ind == P_id
        if P_id==4
        ind = ind - 1;
        else ind = ind + 1;
        end
    end
    
    C_ids(i) = ind;
end

end
    