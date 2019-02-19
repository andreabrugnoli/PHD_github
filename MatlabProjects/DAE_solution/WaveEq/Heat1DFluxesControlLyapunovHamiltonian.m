% 1D heat equation for control purpose (PFEM P1-P1)
% 3 points of view: entropy, internal energy, Lyapunov
% 3 ways to perform simulation: classical parabolic, closed-loop, pHs
%
%=====
%
% Thermo reminder:
%
% Let \rho be the mass density
%
% 1st law: the internal energy of a closed-system is constant
%   \rho dt(u) = -div(Jq), with u the internal energy density and Jq the thermal flux
% 2nd law: the entropy S of a closed system is non-decreasing
% Jaumann's entropy balance:
%   \rho dt(s) = -div(Js) + \sigma, with s is the entropy density and Js := 1/T Jq
%   is the entropy flux and \sigma is the irreversible production of
%   entropy, with T the local temperature defined by du/ds
% Calorimetric equation:
%   Cv := du/dT, with Cv the thermal capacity (constant volume)
% Gibbs formula:
%   dU = T dS => u = Ts, where U is the internal energy
%
% From Fourier's law:
%   Jq := -\lambda.grad(T), with \lambda (tensor) the thermal diffusivity
%   coefficient, we get:
%       \rho Cv dt(T) = \rho du/dT dT/dt = \rho dt(u) = -div(Jq) = div(\lambda.grad(T))
%
%=====
%
% Remark that:
%   div(Js) = -\beta^2 grad(T).Jq + \beta div(Jq)
%
% Hence \rho \beta dt(u) = \rho dt(s) (Gibbs) and then:
%   \rho dt(s) = -\beta div(Jq) = -div(Js) - \beta^2 grad(T).Jq
% So with Fourier's law, \sigma = \beta^2 grad(T).\lambda.\grad(T) and we
% get the Jaumann's entropy balance and retrieve the second law of thermo.
%
%=====
%
% *** Entropy as Hamiltonian: H(u) := S(u) := \int_\Omega s(u) \rho, u as the energy variable
% We have:
%   dt(H) = \int_\Omega ds/du du/dt \rho = \int_\Omega \rho \beta dt(u)
%       = \int_\Omega \rho dt(s) = -\int_\Omega div(Js) + \int_\Omega \sigma
%       = -\int_{\partial\Omega} \beta Jq.n + \int_\Omega \sigma
%
%   du(H) wrt L^2_\rho => ds/du = 1/T =: \beta (we use Gibbs formula)
%   We denote:
%       *- fu := \rho dt(u)
%       *- eu := \beta
%       *- er := Jq (resistive port)
%   Hence, we obtain that fr = -grad(\beta), indeed, this gives the pHs:
%       / fu \ = /   0   -div \ / eu \
%       \ fr /   \ -grad   0  / \ er /
%
%   Equivalently, with thermodynamical variables:
%       /  \rho dt(u)  \ = /   0   -div \ / \beta \
%       \ -grad(\beta) /   \ -grad   0  / \   Jq  /
%
% We then need constitutive laws to link u <-> eu and fr <-> er
%
% *** Internal energy as Hamiltonian: H(s) := U(s) := \int_\Omega u(s) \rho, s as the energy variable
% We have:
%   dt(H) = \int_\Omega du/ds ds/dt \rho = \int_\Omega \rho T dt(s)
%       = \int_\Omega \rho dt(u) = -\int_\Omega div(Jq)
%       = - \int_{\partial\Omega} T Js.n
%
%   ds(H) wrt L^2_\rho => du/ds = T (we use Gibbs formula)
%   We denote:
%       *- fs := \rho dt(s)
%       *- es := T
%       *- er := Js (resistive port)
%   Hence, we obtain that fr = -grad(T), indeed, this gives the pHs:
%       / fs \ = /   0   -div \ / es \ + / \sigma \
%       \ fr /   \ -grad   0  / \ er /   \    0   /
%
%   Equivalently, with thermodynamical variables:
%       / \rho dt(s) \ = /   0   -div \ /  T \ + / \sigma \
%       \  -grad(T)  /   \ -grad   0  / \ Js /   \    0   /
%
% We then need constitutive laws to link s <-> es and fr <-> er
%
% *** Lyapunov as Hamiltonian: H(u) := L(u) := \int_\Omega u^2 \rho / 2 Cv, u as the energy variable
% We have:
%   dt(H) = \int_\Omega du/dt u \rho / Cv
%       = \int_\Omega \rho du/dT dT/dt u / Cv
%       = \int_\Omega \rho dt(T) u
%       = -\int_\Omega div(Jq) u / Cv
%       = \int_\Omega Jq.grad(u / Cv) + \int_{\partial\Omega} (u / Cv) Jq.n
%
%   du(H) wrt L^2_\rho => u / Cv
%   We denote:
%       *- fu := \rho dt(u)
%       *- eu := u / Cv
%       *- er := Jq (resistive port)
%   Hence, we obtain that fr = -grad(u / Cv), indeed, this gives the pHs:
%       / fu \ = /   0   -div \ / eu \
%       \ fr /   \ -grad   0  / \ er /
%
%   Equivalently, with thermodynamical variables:
%       /    \rho dt(u)   \ = /   0   -div \ /  u / Cv \
%       \  -grad(u / Cv)  /   \ -grad   0  / \    Jq   /
%
% We then need constitutive laws to link u <-> eu and fr <-> er:
% *- Using Dulong-Petit => u = Cv T with constant Cv (in time)
% *- Fourier's law => Jq = -\lambda.grad(T)
%   => er = Jq = -\lambda.grad(T) = -\lambda.grad(u / Cv) = \lambda.fr
%
% Hence we also get:
%   H(u) = G(T) = \int_\Omega Cv T^2 / 2
% and
%   dt(H) = -\int_\Omega grad(T).\lambda.grad(T) - \int_{\partial\Omega} T (\lambda.grad(T)).n
%
%=====
%
% The Lyapunov way is tested in this script. 3 ways to simulate it are compared:
% *- classical parabolic
% *- closed-loop pHs which leads to reformulation of the first one
% *- pHs which leads to a DAE
%
% Heat flux boundary control (constant in time for simplicity):
% We impose the boundary controls:
% uL = -Jq(0).n(0) = Jq(0) = -\lambda(0)*dx(T)(0)
%   and
% uR = -Jq(1).n(1) = -Jq(1) = \lambda(1)*dx(T)(1)
%
% Also for simplicity, there is no right-hand side
%

%% Parameters, functions and matrices

% Space
L = 1; % Length
dx = 0.05; % Space step
x = 0:dx:L;
Nx = length(x);

% Time
Tf = 5; % Final time
dt = 0.001; % Time step
t = 0:dt:Tf;
Nt = length(t);

% Physical parameters
Cv = 3.*ones(Nx,1) + 0.5*sin(4*pi*x)'; % Space-dependent heat capacity (at constant volume)
lambda = 0.25*(-x'.*(L-x')+2*L/3); % Space-dependent diffusivity
rho = 3.*ones(Nx,1)-sin(pi*x)'; % Space-dependent mass density
T0 = 10*exp(-10*(x-L/2).^2)' + 10*x'.*x' - 5*x' + 2; % Space-dependent initial temperature
fR0 = -(-100*(2*x'-L).*exp(-10*(x-L/2).^2)' + 20*x' - 5); % Space-dependent -grad(T)

% Controls: boundary heat fluxes (in x=0 and x=L). Constants in these simulations
uL = 1.5; % Left
uR = -2.5; % Right

% All the matrices
MCv = spdiags(Cv, 0, Nx, Nx); % Cv (beware that this is not a mass matrix !)
M = zeros(Nx,Nx); % Mass
MT = zeros(Nx,Nx); % Mass with Cv and rho
Mrho = zeros(Nx,Nx); % Mass with rho
Lambda = zeros(Nx,Nx); % Mass with lambda
D = zeros(Nx,Nx); % "grad"
Aclass = zeros(Nx,Nx); % div(lambda grad)
for i=1:Nx-1
    Melt = (dx/6) * [[2, 1]; [1, 2]];
    M(i:i+1, i:i+1) = M(i:i+1, i:i+1) + Melt;
    MTelt = ((Cv(i)+Cv(i+1))*(rho(i)+rho(i+1))*dx/24) * [[2, 1]; [1, 2]];
    MT(i:i+1, i:i+1) = MT(i:i+1, i:i+1) + MTelt;
    Mrhoelt = ((rho(i)+rho(i+1))*dx/12) * [[2, 1]; [1, 2]];
    Mrho(i:i+1, i:i+1) = Mrho(i:i+1, i:i+1) + Mrhoelt;
    Lambdaelt = ((lambda(i)+lambda(i+1))*dx/12) * [[2, 1]; [1, 2]];
    Lambda(i:i+1, i:i+1) = Lambda(i:i+1, i:i+1) + Lambdaelt;
    Delt = (1/2) * [[-1, -1]; [1, 1]];
    D(i:i+1, i:i+1) = D(i:i+1, i:i+1) + Delt;
    Aelt = ((lambda(i)+lambda(i+1))/(2*dx)) * [[-1, 1]; [1, -1]];
    Aclass(i:i+1, i:i+1) = Aclass(i:i+1, i:i+1) + Aelt;
end

% Sparsing them
M = sparse(M);
MT = sparse(MT);
Mrho = sparse(Mrho);
Lambda = sparse(Lambda);
Aclass = sparse(Aclass);

% Matrix for the "pHs way" (skew-symmetric J)
mDt = -D'; % "-grad"

% Matrix for the "closed-loop way"
Aclosed = D * (M \ Lambda) * (M \ mDt); % div(lambda grad) as a closed-loop

% Boundary-Omega mass matrix
Bnd = sparse(Nx,2);
Bnd(1,1) = 1;
Bnd(Nx,2) = 1;

% For the "pHs way" (skew-symmetric J): boundary ports
Bndt = Bnd';

%% Resolution of the "classical way"
opts = odeset('Mass', MT);
[t, Tclass] = ode45(@(t,Tclass) Aclass*Tclass + Bnd*[uL; uR], t, T0, opts);

%% Resolution of the "closed-loop way"
opts = odeset('Mass', Mrho*MCv);
[t, Tclosed] = ode45(@(t,Tclosed) Aclosed*Tclosed + Bnd*[uL; uR], t, T0, opts);

%% (DAE) Resolution of the "pHs way"
alphaT = sym([]);
eT = sym([]);
fR = sym([]);
eR = sym([]);
syms ts fbL(ts) fbR(ts);
for i=1:Nx
    syms(sprintf('alphaT%d(ts)', i))
    syms(sprintf('eT%d(ts)', i))
    syms(sprintf('fR%d(ts)', i))
    syms(sprintf('eR%d(ts)', i))
    alphaTb = symfun(eval(sprintf('alphaT%d(ts)', i)), ts);
    eTb = symfun(eval(sprintf('eT%d(ts)', i)), ts);
    fRb = symfun(eval(sprintf('fR%d(ts)', i)), ts);
    eRb = symfun(eval(sprintf('eR%d(ts)', i)), ts);
    alphaT = [alphaT; alphaTb];
    eT = [eT; eTb];
    fR = [fR; fRb];
    eR = [eR; eRb];
end
eqn1 = Mrho*diff(alphaT(ts), ts, 1) == D*eR(ts) + Bnd*[uL; uR];
eqn2 = M*fR(ts) == mDt*eT(ts);
eqn3 = M*eR(ts) == Lambda*fR(ts); % Fourier
eqn4 = MCv*eT(ts) == alphaT(ts); % Dulong-Petit
eqn5 = [fbL(ts); fbR(ts)] == Bndt*eT(ts);
eqns = [eqn1; eqn2; eqn3; eqn4; eqn5];
vars = [alphaT(ts); eT(ts); fR(ts); eR(ts); fbL(ts); fbR(ts)];
[DAEeq, DAEvar] = reduceDAEIndex(eqns,vars);
f = daeFunction(DAEeq, DAEvar);
F = @(ts,Y,YP) f(ts,Y,YP);
y0est = [ MCv*T0; T0; fR0; Lambda*fR0; T0(1); T0(Nx) ];
yp0est = [ zeros(Nx,1); zeros(Nx,1); zeros(Nx,1); zeros(Nx,1); 0; 0 ];
[y0, yp0] = decic(F, 0, y0est, [ zeros(Nx,1); ones(Nx,1); zeros(2*Nx,1); 0; 0 ], yp0est, []);
[t,pHs] = ode15i(F, t, y0, yp0);
TpHs = pHs(:,Nx+1:2*Nx);
JqpHs = pHs(:,3*Nx+1:4*Nx);

%% Post-processing and plots

% Matrix to compute the heat fluxes for the "classical and closed-loop ways"
BndFlux = sparse(2,Nx);
BndFlux(1,1) = lambda(1)/dx;
BndFlux(1,2) = -lambda(1)/dx;
BndFlux(2,Nx-1) = -lambda(Nx)/dx;
BndFlux(2,Nx) = lambda(Nx)/dx;

% Lyapunov "int_0^1 rho u^2 / 2 Cv"
Hclass = zeros(Nt,1);
Hclosed = zeros(Nt,1);
HpHs = zeros(Nt,1);

% Internal energy "int_0^1 rho Cv T"
Uclass = zeros(Nt,1);
Uclosed = zeros(Nt,1);
UpHs = zeros(Nt,1);

% Boundary heat fluxes
mJqnclass = zeros(Nt,2);
mJqnclosed = zeros(Nt,2);
mJqnpHs = zeros(Nt,2);

% L^2 absolute errors
Eclassclosed = zeros(Nt,1);
EclasspHs = zeros(Nt,1);
EclosedpHs = zeros(Nt,1);

% Time loop
for i=1:Nt
    Hclass(i) = dx*(Tclass(i,:).^2)*(MCv*rho)/2;
    Hclosed(i) = dx*(Tclosed(i,:).^2)*(MCv*rho)/2;
    HpHs(i) = dx*(TpHs(i,:).^2)*(MCv*rho)/2;
    Uclass(i) = dx*Tclass(i,:)*(MCv*rho);
    Uclosed(i) = dx*Tclosed(i,:)*(MCv*rho);
    UpHs(i) = dx*TpHs(i,:)*(MCv*rho);
    mJqnclass(i,:) = BndFlux*Tclass(i,:)';
    mJqnclosed(i,:) = BndFlux*Tclosed(i,:)';
    mJqnpHs(i,:) = [JqpHs(i,1), -JqpHs(i,Nx)];
    Eclassclosed(i) = dx*(Tclass(i,:)-Tclosed(i,:))*(Tclass(i,:)-Tclosed(i,:))'/2;
    EclasspHs(i) = dx*(Tclass(i,:)-TpHs(i,:))*(Tclass(i,:)-TpHs(i,:))'/2;
    EclosedpHs(i) = dx*(Tclosed(i,:)-TpHs(i,:))*(Tclosed(i,:)-TpHs(i,:))'/2;
end

% Lyapunov and internal energy: classical solution
figure(1)
yyaxis right
plot(t,Hclass)
yyaxis left
plot(t,Uclass)
legend('Internal energy','Lyapunov','Location','Best')
title('Classical solution')
xlabel('t')
grid on

% Lyapunov and internal energy: closed-loop solution
figure(2)
yyaxis right
plot(t,Hclosed)
yyaxis left
plot(t,Uclosed)
legend('Internal energy','Lyapunov','Location','Best')
title('Closed-loop solution')
xlabel('t')
grid on

% Lyapunov and internal energy: pHs solution
figure(3)
yyaxis right
plot(t,HpHs)
yyaxis left
plot(t,UpHs)
legend('Internal energy','Lyapunov','Location','Best')
title('pHs solution')
grid on

% The classical solution in space and time
figure(4)
surf(x,t,Tclass)
title('Classical solution')
xlabel('x')
ylabel('t')
zlabel('T(t,x)')

% The closed-loop solution in space and time
figure(5)
surf(x,t,Tclosed)
title('Closed-loop solution')
xlabel('x')
ylabel('t')
zlabel('T(t,x)')

% The pHs solution in space and time
figure(6)
surf(x,t,TpHs)
title('pHs solution')
xlabel('x')
ylabel('t')
zlabel('T(t,x)')

% Heat fluxes: controls
figure(7)
plot(t,mJqnclass(:,1),t,mJqnclass(:,2),t,mJqnclosed(:,1),t,mJqnclosed(:,2),t,mJqnpHs(:,1),'x',t,mJqnpHs(:,2),'o')
legend('-Left flux (classical)','-Right flux (classical)','-Left flux (closed-loop)','-Right flux (closed-loop)','-Left flux (pHs)','-Right flux (pHs)','Location','Best')
title(strcat(['Boundary controls: -heat fluxes uL = ', num2str(uL), ', uR = ', num2str(uR)]))
xlabel('t')
grid on

% Boundary temperatures: observations
figure(8)
bndTclass = [Tclass(:,1), Tclass(:,Nx)];
bndTclosed = [Tclosed(:,1), Tclosed(:,Nx)];
bndTpHs = [pHs(:,4*Nx+1), pHs(:,4*Nx+2)];
plot(t, bndTclass, t, bndTclosed, t, bndTpHs(:,1), 'x', t, bndTpHs(:,2), 'o')
legend('Left T (classical)','Right T (classical)','Left T (closed-loop)','Right T (closed-loop)','Left T (pHs)','Right T (pHs)','Location','Best')
title('Boundary observations: temperatures')
xlabel('t')
grid on

% L^2 absolute errors
figure(9)
semilogy(t,Eclassclosed,t,EclasspHs,t,EclosedpHs)
legend('Classical vs Closed-loop','Classical vs pHs','Closed-loop vs pHs','Location','Best')
title('Absolute errors between T° (L^2)')
xlabel('t')
grid on
