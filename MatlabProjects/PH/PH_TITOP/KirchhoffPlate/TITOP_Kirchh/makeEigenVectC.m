function [EV_C] = makeEigenVectC(V,P_id,C_ids)

EV_C = zeros(3,length(V(1,:)),length(C_ids));

for i = 1:length(C_ids)
   if C_ids(i) < P_id
       EV_C(:,:,i) = V(3*(C_ids(i)-1)+1:3*(C_ids(i)-1)+3,:);
   else
       EV_C(:,:,i) = V(3*(C_ids(i)-2)+1:3*(C_ids(i)-2)+3,:);
   end
end


end