function R = quat2dcm(q);

if length(q)~=4
    error('The input is not a valid quaternion (length~=4)')
end

% toll= 10^-1;
% if (norm(q)<(1-toll) | norm(q)>(1+toll))
%     error('The input is not a valid quaternion (norm~=1)')
% end

q_0 = q(1);
q_x = q(2);
q_y = q(3);
q_z = q(4);

R = [q_0^2+q_x^2-q_y^2-q_z^2, 2*(q_x*q_y+q_0*q_z), 2*(q_x*q_z-q_0*q_y);
     2*(q_x*q_y-q_0*q_z), q_0^2-q_x^2+q_y^2-q_z^2, 2*(q_y*q_z+q_0*q_x);
     2*(q_x*q_z+q_0*q_y), 2*(q_y*q_z-q_0*q_x), q_0^2-q_x^2-q_y^2+q_z^2];

end