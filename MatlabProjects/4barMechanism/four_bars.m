function [R1,R2] = four_bars( theta2, l1, l2, l3, l4 )
%FOUR_BARS Caluculates the angles between bars in a four-bar closed chain
%   The inputs are the cranck angle and the 4 lmengths of the bars
%   The outputs are two rotation matrices : two possible configurations of
%   the four_bar mechanism
K1 = l1/l2;
K2 = l1/l4;
K3 = (l2^2 - l3^2 + l4^2 +l1^2)/(2*l4*l2);
A = cos(theta2) - K1 - K2*cos(theta2) + K3;
B = -2*sin(theta2);
C = K1 - (K2 + 1)*cos(theta2) + K3;
%there might be different configurations (max 2 or maybe no one!)
if B^2 - 4*A*C < 0
    disp('This configuration is impossible!');
    R1=[];R2=[];
    return
else
    if A ~= 0 
        theta4 = 2*atan((-B - sqrt(B^2 - 4*A*C))/(2*A));
        theta42 = 2*atan((-B + sqrt(B^2 - 4*A*C))/(2*A));
        theta3 = atan((l4*sin(theta4) - l2*sin(theta2))/(l1 - l2*cos(theta2) + l4*cos(theta4)));
        theta32 = atan((l4*sin(theta42) - l2*sin(theta2))/(l1 - l2*cos(theta2) + l4*cos(theta42)));
    else
        theta4 = 2*atan(-C/B);
        theta42 = 0;
        theta3 = atan((l4*sin(theta4) - l2*sin(theta2))/(l1 - l2*cos(theta2) + l4*cos(theta4)));
        theta32 = 0;
    end

    %Matrix R output:
    R1 = zeros(3,3);
    R2 = zeros(3,3);
    a = theta2;
    R1(1,:) = [a 0 0];
    R2(1,:) = [a 0 0];
    b1 = theta3 - theta2;
    b2 = theta32 - theta2;
    % if (b1 > pi/2)|(b2 > pi/2) 
    %     b1 = b1 - pi;
    %     b2 = b2 - pi;
    % end
    R1(2,:) = [b1 0 0];
    R2(2,:) = [b2 0 0];
    c1 = -pi - theta3 + theta4 ;
    c2 = -pi - theta3 + theta42 ;
    % if (c1 > pi/2)|(c2 > pi/2) 
    %     c1 = c1 - pi;
    %     c2 = c2 - pi;
    % end
    R1(3,:) = [c1 0 0];
    R2(3,:) = [c2 0 0];
    
end
end


