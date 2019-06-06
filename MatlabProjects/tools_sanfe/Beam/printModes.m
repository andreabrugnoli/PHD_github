function[] = printModes(nb_mode_shapes, GN_coord, V, D)
% printModes(nb_mode_shapes, GN_coord, V, D)
% 
% Print the first modal shapes.
% Input : 
% nb_mode_shapes : mode shapes number
% GN_coord : global node cordinate matrix
% V : modal deformation matrix
% D : (eigenvalues diagonal matrix).^2


z = [];
z = [z;zeros(1,nb_mode_shapes)];

for i = 1:length(GN_coord)-1
    z = [z; V(2*(i-1)+1,1:nb_mode_shapes)];
end


for i = 1 : nb_mode_shapes
   x = GN_coord(:);
   figure;
   plot(x,z(:,i)); 
   xlabel('x [m]')
   ylabel('z [m]')
   title(['Mode ',num2str(i),': ',num2str(sqrt(D(i,i))./(2*pi)),' Hz'])
   grid on
end


end