clear all
close all
clc
m_panel = 43.2; 
lx=4.143;
ly=2.200;
t=0.04;
rho=m_panel/lx/ly/t;
E=70*10^9;
ni = 0.35;
xi=0.003;

nx = 20;
ny = 20;

P_id=(nx+1)*floor(ny/2)+1   % P est au millieu du petit coté
C_ids=(nx+1)*floor(ny/2)+nx+1

[MtzRyRx] = NPort_KirchhoffTzRyRx(lx, ly, t, rho, E, ni, nx, ny, P_id, C_ids, xi);