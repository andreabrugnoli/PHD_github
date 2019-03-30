clc
close all
clear all

addpath('/home/a.brugnoli/sources/dstools/')

load E; load J; load B; 
% load Etil; load Jtil; load Btil; 
load Er; load Jr; load Br;
% load Er1; load Jr1; load Br1;
% load Er2; load Jr2; load Br2;

sys_full = dss(J,B,B',0,E);
sys_full = minreal(sys_full);
% [isp, sys_full] = isproper(sys_full);

% sys_reg = dss(Jtil,Btil,Btil',0,Etil);
% sys_reg = minreal(sys_reg);
% [isp, sys_red] = isproper(sys_red);

sys_red = dss(Jr,Br,Br',0,Er);
sys_red = minreal(sys_red);

% sys_red1 = dss(Jr1,Br1,Br1',0,Er1);
% sys_red1 = minreal(sys_red1);

% sys_red2 = dss(Jr2,Br2,Br2',0,Er2);
% sys_red2 = minreal(sys_red2);
 
figure(1)
bode(sys_full, sys_red)

% figure(2)
% bode(sys_full, sys_red1)
% figure(3)
% bode(sys_full, sys_red2)


%eig_full = eig(E, J)
%eig_red = eig(Er, Jr)

