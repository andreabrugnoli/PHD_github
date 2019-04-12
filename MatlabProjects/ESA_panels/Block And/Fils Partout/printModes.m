function[] = printModes(nb_mode_shapes,P_id, GN_coord, V, D)

z = [];
flag = 0;
for i = 1:length(GN_coord)
    if i == P_id && flag == 0
        z = [z;zeros(1,nb_mode_shapes)];
        flag = 1;
    elseif flag == 0
        j = i;
        z = [z; V(3*(j-1)+1,1:nb_mode_shapes)];
    else
        j = i-1;
        z = [z; V(3*(j-1)+1,1:nb_mode_shapes)];
    end
end

for i = 1 : nb_mode_shapes
   x = GN_coord(:,1);y=GN_coord(:,2);
   tri = delaunay(x,y); %x,y,z column vectors
   figure;
   trisurf(tri,x,y,z(:,i)); 
   xlabel('x [m]')
   ylabel('y [m]')
   zlabel('z [m]')
   title(['Mode ',num2str(i),': ',num2str(sqrt(D(i,i))./(2*pi)),' Hz'])
   grid on
end


end