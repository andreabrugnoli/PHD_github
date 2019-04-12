% Test beam

E = 2e11;
rho = 7900;  
nu = 0.3;

b = 0.05;
h = 0.01;
S = b * h;

I = 1./12 * b * h^3;

L = 1;


n = 1;
deg = 3;
MtyRz = TwoPort_NElementsBeamTyRz(1, rho, S, L, E, I, 0);

t_ev = linspace(0,1, 1000);
step(MtyRz(3,3), t_ev)
