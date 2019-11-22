%% On colocated control and non-colocated control to achieve performance requirements.

%%
% Fisrt, clean workspace and give access to librairies:
clear all
close all
format short e
bdclose('all')
path(path,'bib1');
path(path,'bib2');
warning('off')
%% Problem statement
%
% Let us considers an Euler bernouilli beam $PC$ (see following Figure) with the 
% following boundary conditions:
%
% * the beam is pinned at $P$ but loaded with a local inertia $I$,
% * the beam is free at $C$.
%
% The revolute joint at $P$ is fitted with an actuator applying a torque
% $u\;(Nm)$ around $(P,\mathbf{z})$ axis.
% 
% <<../beamPC.png>>
%
% *Objectives:* Considering the following disturbances:
%
% * a torque around $(P,\mathbf{z})$-axis: $T^p_{z,./P}\;(Nm)$,
% * a torque around $(C,\mathbf{z})$-axis: $T^p_{z,./C}\,(Nm)$,
% * a force along $(C,\mathbf{y})$-axis: $F^p_{y,./C}\;(N)$,
%
% the objective is to reject (in all the frequency-domain) these disturbances on the angular 
% acceleration $\ddot{\theta}_C$ at $C$ (assuming here is a massless optical payload at $C$).
% 
% Thus the performance index is defined as:
%
% $$\|\mathrm{T_f}_{\left[\begin{array}{c}F^p_{y,./C}\\ T^p_{z,./C} \\ T^p_{z,./P} \end{array}\right]\to\ddot{\theta}_C}(\mathrm{s})\|_\infty=\max_{\omega}\bar{\sigma}\Big(\mathrm{T_f}_{\left[\begin{array}{c}F^p_{y,./C}\\ T^p_{z,./C} \\ T^p_{z,./P} \end{array}\right]\to\ddot{\theta}_C}(\mathrm{j\omega})\Big)\;.$$
%
%% Dynamic Model
%
% Applying the TITOP (Two-Input Two-Output Port) model approach and the
% channel inversion operation (see also: <http://oatao.univ-toulouse.fr/16559/  MBS>, _Linear dynamics of flexible
% multibody systems : a system-based approach_, C. Jawhar _t al._), the
% design model $\mathbf G(\mathrm{s})$, depicted in the following Figure can be
% easily derived by the following procedure:
%
% * first, compute the cantilever-free model of the beam using
%   one "superlement" thanks to the function |TwoPortBeamTyRz| (see the
%   |help|). All the parameters of this academic example are normalize to
%   $1$. A damping ratio ($0.003$) is chosen for the flexible modes:

L=1;  % Length of the beam
Mod=TwoPortBeamTyRz(1,1,L,1,1,0.003);

%%
% * then inverse the last channel to model pinned-free boundary conditions:

Mpf=invio(Mod,4);

%%
% * then feedback the local inertia $I$ at point $P$:

I=1;   

%%
% The model $\mathbf{G}(\mathrm{s})$:
open_system('OLmodel','force')

[a,b,c,d]=linmod('OLmodel');
G=ss(a,b,c,d);

%% 
%This model if a 8-th order model with 4 flexible modes:
damp(G)

%% Colocated control
% It is well-known that the control law $u=-K_v\dot{\theta}_p$ is
% stabilizing for any positive value of $K_v$. Although increasing $K_v$ up
% to $+\infty$ would allow to reject perfectly the disturbance
% $T^p_{z,./P}$, the rejection of distrubances from $F^p_{y,./C}$ and
% $T^p_{z,./C}$ cannot be improved by such a colocated control and is limited
% by the open-loop rejection properties of the cantilevered-free beam.
%
% That is highlighted by the following analyse (in the case $K_v=-3\;Nms/rd)$.

CLco=feedback(G,3*tf(1,[1 0]),4,4);
figure
sigma(G(2,[1:3]),CLco(2,[1:3]),Mod(2,[1:2]))
title('Disturbance rejection')
legend('open-loop','colocated control','cantilever-free beam','Location','southeast');
%%
% Performance indexes:
norm(G(2,[1:3]),'inf')    % open-loop
%%
norm(CLco(2,[1:3]),'inf') % colocated control
%%
norm(Mod(2,[1:2]),'inf')  %  cantilever-free beam.

%% Noncolocated control
% A non colacted control law $u=-K  \dot{\theta_C}$ cannot be used to increase the damping 
% of all the flexible modes. Indeed:
figure
rlocusp(tf(1,[1 0])*G(2,4));
axis([-10 10 -10 400])
%% 
% A dynamic controller $K\to K(\mathrm{s})$ is thus required and its design is
% a complex task.
%
% Now we assume that an angular accelerometer can measure
% $\ddot{\theta}_C$. We propose a simple procedure to design a dynamic
% controller allowing to damp all the flexible modes and thus to increase the
% disturbance rejection property. Such a control law is depicted in the
% following block diagram. It involves:
%
% * a gain $I_{f,C}\;(Kg\,m^2)$ corresponding to the fictitious inertia
% to be included at point $C$ to increase disturbance rejection,
% * a reference model $G_{ref}(\mathrm{s})$ between the desired torque $T_{z,./C}^{ref}=-I_{f,C}\ddot{\theta}_C$ at $C$ and the 
% desired acceleration $\ddot{\theta}_P^{ref}$ at $P$ . This reference model corresponds to the transfer
% from $T_{z,./C}$ to $\ddot{\theta}_P$, i.e. $\mathbf{G}_{(4,2)}(\mathrm{s})$
% where the flexible mode damping ratio is prescribed to a given value
% $\xi_{ref}$,
% * the total inertia $I_{tot,P}\;(Kg\,m^2)$ of the system seen from the point $P$ to
% transform the desired angular
% acceleration $\ddot{\theta}_P^{ref}$ at $P$ to the torque $u=-I_{tot,P}\ddot{\theta}_P^{ref}$ to be applied
% by the actuator.
%
% This control law has 2 tuning parameters: $I_{f,C}$ and $\xi_{ref}$. In
% the following sequence: $I_{f,C}=0.05\,I_{tot,P}$ and $\xi_{ref}=0.3$.
I_tot_P=inv(dcgain(G(4,4)));
I_f_C=0.05*I_tot_P;
Gref=G(4,2);             % Transfer from Tz_./C to ddot_theta_P
[V,D]=eig(Gref.a(5:8,1:4));
xi_ref=0.3; 
Gref.a(5:8,5:8)=V*(-2*xi_ref*sqrt(-D))*inv(V);

%
% Root locus:
figure
rlocusp(-I_tot_P*I_f_C*Gref*G(2,4))
axis([-100 10 -10 400])

%%
% Closed-loop model:
open_system('CLmodel','force')

%%
[a,b,c,d]=linmod('CLmodel');
CLnoc=ss(a,b,c,d);

%%
% Disturbance rejection:
figure
sigma(G(2,[1:3]),CLco(2,[1:3]),CLnoc(2,[1:3]))
title('Disturbance rejection')
legend('open-loop','colocated control','non-colocated control','Location','southeast');
%%
% Performance index:
norm(CLnoc(2,[1:3]),'inf')

%%
% Thus the perfornance index of the non-colocated control is
% significantly better than the colocated control's one.

%% Questions and remarks:
%
% * *Q1:* stability proof of such an approach in the infinite dimension case ?
% * *Q2:* tuning of the 2 parameters: $I_{f,C}$ and $\xi_{ref}$ ?
% * *R1:* the same approach could be used if an accelerometer measures
% $\ddot y_c$. Then the reference model is $1\times 2$ built from
% $\mathbf{G}_{(4,1:2)}(\mathrm{s})$.

%% SPILLOVER
% From a practical point of view and beyond the stability proof in the
% infinite-dimension case, such a controller must be designed from a finite
% order reference model $G_{ref}(\mathrm{s})$. Therfore, the problem of
% spillover (i.e. the stability of flexible modes which are taken into
% account in $G_{ref}(\mathrm{s})$) must be addressed. So let us a consider
% a 16-th order validation model $G_f(\mathrm{s})$ composed 2 superelements of half
% lenght:
Modval=TwoPortBeamTyRz(1,1,L/2,1,1,0.003);
Modval=lft(Modval,Modval,2,2);
Mpf=invio(Modval,4);
[a,b,c,d]=linmod('OLmodel');
Gf=ss(a,b,c,d);

%%
% The previous controller is not stabilizing: the 5-th flexible mode
% of the validation model $G_f$ is unstable. Indeed:
figure
rlocusp(-I_tot_P*I_f_C*Gref*Gf(2,4))
axis([-100 10 -10 400])

%%
% So we propose to build $G_{ref}(\mathrm{s})$ from a 8-th order reduced
% model $G_r(\mathrm{s})$. This model is obtained from a modal reduction of
% $G_f(\mathrm{s})$ to catch the first 4 low frequency flexible modes.
% That can be done using the function |red_fast.m| (see the |help|).
damp(Gf)
Gr=red_fast(Gf,-1);

%%
% Reference model $G_{ref}(\mathrm{s})$:
Gref=Gr(4,2);

%%
% The damping ratio $\xi_{ref}$ is prescribed in the modal representation
% of $G_r(\mathrm{s})$:
for ii=1:length(Gref.a)/2,
    wii=damp(Gref.a(2*ii-1:2*ii,2*ii-1:2*ii));
    wii=wii(1);
    Gref.a(2*ii-1:2*ii,2*ii-1:2*ii)=[-xi_ref sqrt(1-xi_ref^2);-sqrt(1-xi_ref^2) -xi_ref]*wii;
end
damp(Gref)

%%
% Validation on the model $G_f$:
figure
rlocusp(-I_tot_P*I_f_C*Gref*Gf(2,4))
axis([-100 10 -10 1500])

%%
% The closed-loop system is stable. One can also illustrate that:
%
% * the 4 first flexible modes are phase-controlled,
% * the 4 last flexible modes are gain-controlled
%
% on the Nichols plot:
figure
nichols(-I_tot_P*I_f_C*Gref*Gf(2,4))
ngrid

%%
% From the spillover point of view, better results are obtained by
% cancelling the direct feedthrough in $G_{ref}$:
Gref=ss(Gref.a,Gref.b,Gref.c,0);
figure
nichols(-I_tot_P*I_f_C*Gref*Gf(2,4))
ngrid

%%
% Closed-loop model:
[a,b,c,d]=linmod('CLmodel');
CLnoc=ss(a,b,c,d);

%%
% Disturbance rejection:
CLco=feedback(Gf,3*tf(1,[1 0]),4,4);
figure
sigma(Gf(2,[1:3]),CLco(2,[1:3]),CLnoc(2,[1:3]))
title('Disturbance rejection')
legend('open-loop','colocated control','non-colocated control','Location','southeast');

%%
% The high frequency flexible modes, which are not controlled, degrade
% consequently the performance. The conclusion is : the non-collocated
% control has better performance rejection than the colocated control in
% the $[0 200]\;(rad/s)$ frequency-band.

%% Colacated-noncolocated control
% If we consider the distrubance rejection on $\theta_C$ (instead of
% $\ddot{\theta}_C$), the colocated control (for the rigid mode) and the
% noncolocated control (for the flexible modes) can be mixed according to
% the following control structure:
Kv=3;Kp=3.5;
open_system('CLmodelMix','force')

%%
% Closed-loop model:
[a,b,c,d]=linmod('CLmodelMix');
CL=minreal(ss(a,b,c,d));

%%
% Closed-loop model in the pure colocated control case
I_f_C=0;
[a,b,c,d]=linmod('CLmodelMix');
CLco=minreal(ss(a,b,c,d));

%%
% Disturbance rejection:
figure
sigma(tf(1,[1 0 0])*Gf(2,[1:3]),CLco(2,[1:3]),CL(2,[1:3]))
grid
title('Disturbance rejection')
legend('open-loop','colocated only control','colocated-noncolocated control','Location','southeast');

%%
% Performance indexes:
norm(CLco(2,[1:3]),'inf')
norm(CL(2,[1:3]),'inf')

%%
% *From the performance index, the benefit of the non-colocated loop is
% quite obvious (a factor 10 is won on the performance index).*
