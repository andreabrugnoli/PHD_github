function R_z = rot_z(angle_rad)

R_z = [cos(angle_rad), -sin(angle_rad), 0;
       sin(angle_rad), cos(angle_rad),  0;
       0,              0             ,  1];