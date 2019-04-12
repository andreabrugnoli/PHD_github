function U_C = evalShapeMatrix(GN_coord, Cs_xy, P_id)

[n_free, ~] = size(Cs_xy);
U_C = zeros(3,9,n_free);

px = GN_coord(:,1);
py = GN_coord(:,2);

for j = 1:n_free

    xC = Cs_xy(j,1);
    yC = Cs_xy(j,2);
    
    
    P = [1 xC yC xC^2 xC*yC yC^2 xC^3 xC^2*yC xC*yC^2 yC^3 xC^3*yC xC*yC^3; ...
          0 0 1 0 xC 2*yC 0 xC^2 2*xC*yC 3*yC^2 xC^3 3*xC*yC^2; ...
         0 -1 0 -2*xC -yC 0 -3*xC^2 -2*xC*yC -yC^2 0 -3*xC^2*yC -yC^3];

    C = zeros(12,12);
    for i = 1:4
       C((3*i-2):3*i,:) = [1 px(i) py(i) px(i)^2 px(i)*py(i) py(i)^2 px(i)^3 px(i)^2*py(i) px(i)*py(i)^2 py(i)^3 px(i)^3*py(i) px(i)*py(i)^3; ...
         0 0 1 0 px(i) 2*py(i) 0 px(i)^2 2*px(i)*py(i) 3*py(i)^2 px(i)^3 3*px(i)*py(i)^2; ...
         0 -1 0 -2*px(i) -py(i) 0 -3*px(i)^2 -2*px(i)*py(i) -py(i)^2 0 -3*px(i)^2*py(i) -py(i)^3];
    end
    
    N = P*inv(C);
    
    U_C(:,:,j) = [N(:,1:3*(P_id-1)) N(:,3*(P_id-1)+4:12)];

end

end