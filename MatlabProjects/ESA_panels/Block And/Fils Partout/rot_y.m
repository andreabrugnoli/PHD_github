function R_y = rot_y(angle_rad)

R_y = [cos(angle_rad), 0, sin(angle_rad);
       0,              1, 0;
       -sin(angle_rad) 0  cos(angle_rad)];