rho1 = 0.2;  % kg/m
EI1 = 1;  % N m^2
L1 = 0.5;  % m
m_joint1 = 0.5;
J_joint1 = 0.1;% kg/m^2

rho2 = 0.2;  % kg/m
EI2 = 1; % N m^2
L2 = 0.5; % m
J_joint2 = 0.1; % kg/m^2
m_joint2 = 1;

m_payload = 0.1;  % kg
J_payload = 0.5 * 10^(-3);  % kg/m^2

w0 = 1e-5;
wf = 1e6;
kp1 = 160;
kv1 = 11;
kp2 = 60;
kv2 = 1.1;
alpha_rel = 60*pi/180;
rad_deg = 180/pi;
