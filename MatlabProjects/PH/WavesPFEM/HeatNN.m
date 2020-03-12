%% 1D heat equation for control purpose (PFEM P1-P1), Lyapunov
% 3 ways to perform simulation: classical parabolic, pHs + Substitution,
%   and pH descriptor

%% Parameters and functions

% Space
Nx = 51; % Number of points
dx = 1/(Nx-1);
x = 0:dx:1;

% Final time
Tf = 25; 

% Physical parameters
cv = 3.*ones(Nx,1) + 0.05*sin(4*pi*x)'; % Space-dependent heat capacity (at constant volume)
lambda = 1.25*(-x'.*(1-x')+2/3); % Space-dependent diffusivity
rho = 10.*ones(Nx,1)-cos(pi*x)'; % Space-dependent mass density
T0 = 20*exp(-10*(x-1/2).^2)' + 5*x'.*x' - 8*x' + 2; % Space-dependent initial temperature

% Controls: boundary heat fluxes
uL = @(t) - cos(t/4) .* exp(-0.25*t) * lambda(1) * (T0(2)-T0(1)) / dx; % Left
uR = @(t) - cos(t/2) .* exp(-0.25*t) * lambda(Nx) * (T0(Nx)-T0(Nx-1)) / dx; % Right

%% Matrices

% All the matrices
M = zeros(Nx,Nx); % "Usual" mass
Cv = zeros(Nx,Nx); % Mass with cv
Rho = zeros(Nx,Nx); % Mass with rho
CR = zeros(Nx,Nx); % Mass with cv*rho
Lambda = zeros(Nx,Nx); % Mass with lambda
Lambdam = zeros(Nx,Nx); % Mass with lambda^{-1} /!\ WARNING /!\ for efficiency of the DAE solver => - Dt * T = Lambda^{-1} * Jq
D = zeros(Nx,Nx); % "div"
Aclass = zeros(Nx,Nx); % div(lambda grad)
Melt = [[2, 1]; [1, 2]];
Delt = (1/2) * [[-1, -1]; [1, 1]];
for i=1:Nx-1
    M(i:i+1, i:i+1) = M(i:i+1, i:i+1) + (dx/6) * Melt;
    CRelt = ((cv(i)+cv(i+1))*(rho(i)+rho(i+1))*dx/24) * Melt;
    CR(i:i+1, i:i+1) = CR(i:i+1, i:i+1) + CRelt;
    Lambdaelt = ((lambda(i)+lambda(i+1))*dx/12) * Melt;
    Lambda(i:i+1, i:i+1) = Lambda(i:i+1, i:i+1) + Lambdaelt;
    Lambdamelt = (dx/(3*(lambda(i)+lambda(i+1)))) * Melt; % /!\ WARNING /!\ Lambda^{-1}
    Lambdam(i:i+1, i:i+1) = Lambdam(i:i+1, i:i+1) + Lambdamelt;
    D(i:i+1, i:i+1) = D(i:i+1, i:i+1) + Delt;
    Aelt = ((lambda(i)+lambda(i+1))/(2*dx)) * [[-1, 1]; [1, -1]];
    Aclass(i:i+1, i:i+1) = Aclass(i:i+1, i:i+1) + Aelt;
end
% Sparsing them
M = sparse(M);
Rho = sparse(Rho);
CR = sparse(CR);
Lambda = sparse(Lambda);
Lambdam = sparse(Lambdam);
D = sparse(D);
Aclass = sparse(Aclass);

% For the "pHs way"
Z = zeros(Nx,Nx);
E = [[CR, Z]; [Z, Z]]; % Singular mass matrix

% Matrix for the "substitution way"
Asubs = D * (M \ Lambda) * (M \ -D'); % div(lambda grad) using substitution

% Boundary-Omega mass matrix
Bnd = sparse(Nx,2);
Bnd(1,1) = 1;
Bnd(Nx,2) = -1;

%% Resolution of the "classical way"
disp('====================')
disp('Solve classical way:')
disp('====================')
opts = odeset('Mass', CR, 'MassSingular', 'no', 'Stats','on');
tic
[tclass, Tclass] = ode23t(@(t, Tclass) Aclass*Tclass + Bnd*[uL(t); uR(t)], [0, Tf], T0, opts);
toc

%% Resolution of the "substitution way"
disp('=======================')
disp('Solve substitution way:')
disp('=======================')
opts = odeset('Mass', CR, 'MassSingular', 'no', 'Stats','on');
tic
[tsubs, Tsubs] = ode23t(@(t, Tsubs) Asubs*Tsubs + Bnd*[uL(t); uR(t)], [0, Tf], T0, opts);
toc

%% (DAE) Resolution of the "pHs way"
% See the file HeatNNpHs.m
disp('==============')
disp('Solve pHs way:')
disp('==============')
opts = odeset('Mass', E, 'MassSingular', 'yes', 'Stats','on');
der = ones(Nx,1);
der([1,Nx]) = 2;
alpha0 = [ T0 ; - der .* lambda .* (D' * T0) / dx ]; % -der*D'/dx = discretization of the gradient, Jq0 := - lambda grad(T0)
tic
[tpHs, alpha] = ode23t(@(t, alpha) HeatNNpHs(t, alpha, D, Bnd, Lambdam, uL, uR, Nx), [0, Tf], alpha0, opts);
toc
TpHs = alpha(:,1:Nx);
JqpHs = alpha(:,Nx+1:2*Nx);

%% Post-processing and plots

% Times
Ntclass = length(tclass);
Ntsubs = length(tsubs);
NtpHs = length(tpHs);

% One vector for the internal energy
OneVect = ones(1,Nx);

% Heat fluxes
Jqclass = - der .* lambda .* (D' * Tclass') / dx; % -der*D'/dx = discretization of the gradient
Jqsubs = - der .* lambda .* (D' * Tsubs') / dx; % -der*D'/dx = discretization of the gradient

% Lyapunov "int_0^1 rho u^2 / 2 Cv"
Hclass = zeros(Ntclass,1);
Hsubs = zeros(Ntsubs,1);
HpHs = zeros(NtpHs,1);

% Internal energy "int_0^1 rho Cv T"
Uclass = zeros(Ntclass,1);
Usubs = zeros(Ntsubs,1);
UpHs = zeros(NtpHs,1);

% Boundary heat fluxes
mJqnclass = zeros(Ntclass,2);
mJqnsubs = zeros(Ntsubs,2);
mJqnpHs = zeros(NtpHs,2);

% Time loop
for i=1:Ntclass
    Hclass(i) = Tclass(i,:)*CR*Tclass(i,:)'/2;
    Uclass(i) = OneVect*CR*Tclass(i,:)';
    mJqnclass(i,:) = Bnd' * Jqclass(:,i);
end
for i=1:Ntsubs
    Hsubs(i) = Tsubs(i,:)*CR*Tsubs(i,:)'/2;
    Usubs(i) = OneVect*CR*Tsubs(i,:)';
    mJqnsubs(i,:) = Bnd' * Jqsubs(:,i);
end
for i=1:NtpHs
    HpHs(i) = TpHs(i,:)*CR*TpHs(i,:)'/2;
    UpHs(i) = OneVect*CR*TpHs(i,:)';
    mJqnpHs(i,:) = Bnd' * JqpHs(i,:)';
end

% Lyapunov energy
figure(1)
plot(tclass,Hclass,tsubs,Hsubs,tpHs,HpHs)
ylabel('Hamiltonian Lyapunov (J)')
xlabel('t (s)')
title('"Lyapunov" energy')
legend('Lyapunov (classical)','Lyapunov (substitution)','Lyapunov (pHs)','Location','Best')
grid on

% Internal energy
figure(2)
plot(tclass,Uclass,tsubs,Usubs,tpHs,UpHs)
ylabel('Internal energy (J)')
xlabel('t (s)')
title('Internal energies')
legend('Internal (classical)','Internal (substitution)','Internal (pHs)','Location','Best')
grid on

% The classical solution in space and time
figure(3)
surf(x,tclass,Tclass,'EdgeColor','none')
title('Classical solution')
xlabel('x (m)')
ylabel('t (s)')
zlabel('T(t,x) (°)')

% The substitution solution in space and time
figure(4)
surf(x,tsubs,Tsubs,'EdgeColor','none')
title('Substitution solution')
xlabel('x (m)')
ylabel('t (s)')
zlabel('T(t,x) (K)')

% The pHs solution in space and time
figure(5)
surf(x,tpHs,TpHs,'EdgeColor','none')
title('pHs solution')
xlabel('x (m)')
ylabel('t (s)')
zlabel('T(t,x) (K)')

% Heat fluxes: controls
figure(6)
tspan=0:(Tf/26):Tf;
plot(tclass,mJqnclass(:,1),tclass,mJqnclass(:,2),tsubs,mJqnsubs(:,1),tsubs,mJqnsubs(:,2),...
    tpHs,mJqnpHs(:,1),tpHs,mJqnpHs(:,2),tspan,uL(tspan),'x',tspan,-uR(tspan),'x')
legend('-Left flux (classical)','-Right flux (classical)','-Left flux (substitution)','-Right flux (substitution)',...
    '-Left flux (pHs)','-Right flux (pHs)','-Left flux (exact)','-Right flux (exact)','Location','Best')
title('Boundary controls')
xlabel('t (s)')
ylabel('Jq.n (W / m^2)')
grid on

% Boundary temperatures: observations
figure(7)
bndTclass = [Tclass(:,1), Tclass(:,Nx)];
bndTsubs = [Tsubs(:,1), Tsubs(:,Nx)];
bndTpHs = [TpHs(:,1), TpHs(:,Nx)];
plot(tclass, bndTclass, tsubs, bndTsubs, tpHs, bndTpHs)
legend('Left T (classical)','Right T (classical)','Left T (substitution)','Right T (substitution)','Left T (pHs)','Right T (pHs)','Location','Best')
title('Boundary observations: temperatures')
xlabel('t (s)')
ylabel('T (K)')
grid on
