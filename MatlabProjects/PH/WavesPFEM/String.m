%% 1D vibrating string equation for control purpose (PFEM P1-P1)
% 3 causalities: force controls, velocity controls, and Mixed

%% Parameters and functions

% Space
Nx = 101; % Number of points
dx = 1/(Nx-1);
x = 0:dx:1;

% Final time
Tf = 1; 

% Physical parameters
rho = 1.*ones(Nx,1); % Space-dependent mass density
T = 1.*ones(Nx,1); % Space-dependent Young's modulus
w0 = -5*sin(2*pi*x)'.*cos(2*pi*x)'; % Space-dependent initial deflection
dtw0 = 2*sin(pi*x/2)'.*sin(pi*x/2)'; % Space-dependent initial deflection velocity

% Controls: boundary forces (Neumann)
uFL = @(t) cos(pi*t) .* T(1) * (w0(2)-w0(1)) / dx; % Left
uFR = @(t) cos(4*pi*t) .* T(Nx) * (w0(Nx)-w0(Nx-1)) / dx; % Right

% Controls: boundary velocities (Dirichlet)
uVL = @(t) cos(pi*t) .* dtw0(1); % Left
uVR = @(t) cos(4*pi*t) .* dtw0(Nx); % Right

%% Matrices

% All the matrices /!\ WARNING /!\ co-energy => T^{-1} and rho !!!
MRho = zeros(Nx,Nx); % Mass with rho
MT = zeros(Nx,Nx); % Mass with T^{-1}
D = zeros(Nx,Nx); % "div" <= for force controls
G = zeros(Nx,Nx); % "grad" <= for velocity controls
Melt = [[2, 1]; [1, 2]];
Delt = (1/2) * [[1, 1]; [-1, -1]];
Gelt = (1/2) * [[1, 1]; [-1, -1]];
for i=1:Nx-1
    MRhoelt = ((rho(i)+rho(i+1))*dx/12) * Melt;
    MRho(i:i+1, i:i+1) = MRho(i:i+1, i:i+1) + MRhoelt; % /!\ WARNING /!\ rho
    MTelt = (dx/(3*(T(i)+T(i+1)))) * Melt; % /!\ WARNING /!\ T^{-1}
    MT(i:i+1, i:i+1) = MT(i:i+1, i:i+1) + MTelt;
    D(i:i+1, i:i+1) = D(i:i+1, i:i+1) + Delt;
    G(i:i+1, i:i+1) = G(i:i+1, i:i+1) + Gelt;
end
% Sparsing them
MRho = sparse(MRho);
MT = sparse(MT);
D = sparse(D);
G = sparse(G);

% "Total" mass for q- and p-type variables
Z = zeros(Nx,Nx);
Mtot = [[MT, Z]; [Z, MRho]];

% Structure J matrix force controls
JFC = [[Z, -D']; [D, Z]]; % /!\ WARNING /!\ D is computed on the SECOND line!

% Boundary force controls mass matrix
BFC = sparse(2*Nx,2);
BFC(Nx+1,1) = -1;
BFC(2*Nx,2) = 1;

% Structure J matrix velocity controls
JVC = [[Z, G]; [-G', Z]];

% Boundary velocity controls mass matrix
BVC = sparse(2*Nx,2);
BVC(1,1) = 1;
BVC(Nx,2) = 1;

%% Compatible nitial co-energy variables
der = ones(Nx,1);
der([1,Nx]) = 2;
eq0 = der .* T .* (G * w0) / dx; % der.*G/dx = discretization of the gradient
ep0 = dtw0;
e0 = [eq0; ep0];

%% Resolution of the force controls causality
disp('=====================')
disp('Solve force controls:')
disp('=====================')
opts = odeset('Mass', Mtot, 'MassSingular', 'no', 'Stats','on');
tic
[tFC, eFC] = ode23t(@(t, eFC) JFC*eFC + BFC*[uFL(t); uFR(t)], [0, Tf], e0, opts);
toc

%% Resolution of the velocity controls causality
disp('========================')
disp('Solve velocity controls:')
disp('========================')
opts = odeset('Mass', Mtot, 'MassSingular', 'no', 'Stats','on');
tic
[tVC, eVC] = ode23t(@(t, eVC) JVC*eVC + BVC*[uVL(t); uVR(t)], [0, Tf], e0, opts);
toc

% %% Resolution of the mixed controls causality
% disp('=====================')
% disp('Solve mixed controls:')
% disp('=====================')
% opts = odeset('Mass', Mtot, 'MassSingular', 'yes', 'Stats','on');
% tic
% [tVC, eVC] = ode23t(@(t, eVC) JVC*eVC + BVC*[uVL(t); uVR(t)], [0, Tf], e0, opts);
% toc

%% Post-processing

% Times
NtFC = length(tFC);
NtVC = length(tVC);
%NtDN = length(tDN);

% Hamiltonian
HFC = zeros(NtFC,1);
HFC(1) = e0'*Mtot*e0/2;
HVC = zeros(NtVC,1);
HVC(1) = e0'*Mtot*e0/2;
%HDN = zeros(NtDN,1);
%HDN(1) = e0'*Mtot*e0/2;

% Deflections
wFC = zeros(NtFC,Nx);
wFC(1,:) = w0;
wVC = zeros(NtVC,Nx);
wVC(1,:) = w0;
%wDN = zeros(NtDN,Nx);
%wDN(1,:) = w0;

% Time loop
for i=2:NtFC
    HFC(i) = eFC(i,:)*Mtot*eFC(i,:)'/2;
    wFC(i,:) = wFC(i-1,:) + (tFC(i)-tFC(i-1)) * eFC(i,Nx+1:2*Nx);
end
for i=2:NtVC
    HVC(i) = eVC(i,:)*Mtot*eVC(i,:)'/2;
    wVC(i,:) = wVC(i-1,:) + (tVC(i)-tVC(i-1)) * eVC(i,Nx+1:2*Nx);
end
%for i=2:NtDN
%    HDN(i) = eDN(i,1:2*Nx)*Mtot*eDN(i,1:2*Nx)'/2;
%    wDN(i,:) = wDN(i-1) + (tDN(i)-tDN(i-1)) * (eDN(i-1,Nx+1:2*Nx) + eDN(i,Nx+1:2*Nx))/2;
%end

% Outputs
yFC = BFC'*eFC';
yVC = BVC'*eVC';

% Power balance
dotHFCext = eFC(:,1).*yFC(1,:)' + eFC(:,Nx).*yFC(2,:)'; % Computed with inputs--outputs
dotHFCint = ((HFC(3:end)-HFC(2:end-1))./(tFC(3:end)-tFC(2:end-1))); % Computed with HFC
dotHVCext = eVC(:,Nx+1).*yVC(1,:)' + eVC(:,2*Nx).*yVC(2,:)'; % Computed with inputs--outputs
dotHVCint = ((HVC(3:end)-HVC(2:end-1))./(tVC(3:end)-tVC(2:end-1))); % Computed with HVC
% dotHVCext = eVC(:,Nx+1).*yVC(1,:)' + eVC(:,2*Nx).*yVC(2,:)'; % Computed with inputs--outputs
% dotHVCint = ((HVC(3:end)-HVC(2:end-1))./(tVC(3:end)-tVC(2:end-1))); % Computed with HVC

%% Displays

% Hamiltonians
figure(1)
plot(tFC,HFC,tVC,HVC)%,tpHs,HDN)
ylabel('Energiy (J)')
xlabel('t (s)')
title('Hamiltonians')
legend('Force controls','Velocity controls','Location','Best')%,'Mixed'
grid on

% The force controls solution in space and time
figure(2)
surf(x,tFC,wFC,'EdgeColor','none')
title('Force controls solution (Deflection)')
xlabel('x (m)')
ylabel('t (s)')
zlabel('w(t,x) (m)')

% The velocity controls solution in space and time
figure(3)
surf(x,tVC,wVC,'EdgeColor','none')
title('Velocity controls solution (Deflection)')
xlabel('x (m)')
ylabel('t (s)')
zlabel('w(t,x) (m)')

% % The mixed solution in space and time
% figure(4)
% surf(x,tDN,wDN,'EdgeColor','none')
% title('Mixed controls solution')
% xlabel('x (meters)')
% ylabel('t (seconds)')
% zlabel('w(t,x) (millimeters)')

% Boundary force controls
figure(5)
tspan=0:(Tf/26):Tf;
plot(tFC,-eFC(:,1),'-b',tFC,eFC(:,Nx),'-r',tspan,-uFL(tspan),'ob',tspan,uFR(tspan),'xr')
legend('-Left force','Right force','-Left force (exact)','Right force (exact)',...
    'Location','Best')
title('Boundary force controls')
xlabel('t (s)')
ylabel('Force (N)')
grid on

% Boundary velocity controls
figure(6)
plot(tVC,eVC(:,Nx+1),'-b',tVC,eVC(:,2*Nx),'-r',tspan,uVL(tspan),'ob',tspan,uVR(tspan),'xr')
legend('Left velocity','Right velocity','Left velocity (exact)','Right velocity (exact)',...
    'Location','Best')
title('Boundary velocity controls')
xlabel('t (s)')
ylabel('Velocity (m/s)')
grid on

% % Boundary mixed controls
% figure(7)
% plot(tMC,eMC(:,Nx+1),'-b',tMC,eMC(:,Nx),'-r',tspan,uVL(tspan),'ob',tspan,uFR(tspan),'xr')
% legend('Left velocity','Right force','Left velocity (exact)','Right force (exact)',...
%     'Location','Best')
% title('Boundary mixed controls')
% xlabel('t (s)')
% ylabel('Velocity (m/s) & Force (N)')
% grid on

% Boundary force observations
figure(8)
plot(tFC,yFC(1,:),tFC,yFC(2,:))
legend('Left velocity','Right velocity','Location','Best')
title('Boundary observations, collocated to force controls')
xlabel('t (s)')
ylabel('Observed velocities (m/s), force controls')
grid on

% Boundary force observations
figure(9)
plot(tVC,yVC(1,:),tVC,yVC(2,:))
legend('-Left force','Right force','Location','Best')
title('Boundary observations, collocated to velocity controls')
xlabel('t (s)')
ylabel('Observed forces (N), velocity controls')
grid on

% % Boundary mixed observations
% figure(10)
% plot(tMC,yMC(1,:),tMC,yMC(2,:))
% legend('-Left force','Right velocity','Location','Best')
% title('Boundary observations, collocated to mixed controls')
% xlabel('t (s)')
% ylabel('Observed force (N) & Observed velocity (m/s)')
% grid on

% Power Balances for force controls
figure(11)
plot(tFC,dotHFCext,tFC(2:end-1),dotHFCint)
legend('Force controls (external)','Force controls (internal)',...
    'Location','Best')
title('Power balances for force controls')
xlabel('t (s)')
ylabel('Power (W)')
grid on

% Power Balances for velocity controls
figure(12)
plot(tVC,dotHVCext,tVC(2:end-1),dotHVCint)
legend('Velocity controls (external)','Velocity controls (internal)',...
    'Location','Best')
title('Power balances for velocity controls')
xlabel('t (s)')
ylabel('Power (W)')
grid on

% % Power Balances for mixed controls
% figure(13)
% plot(tMC,dotHMCext,tMC(2:end-1),dotHMCint)
% legend('Mixed controls (external)','Mixed controls (internal)',...
%     'Location','Best')
% title('Power balances for mixed controls')
% xlabel('t (s)')
% ylabel('Power (W)')
% grid on
