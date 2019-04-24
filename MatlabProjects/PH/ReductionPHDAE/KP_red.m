clc
close all
clear all

addpath('./KP_Matrices/')

load M; load J; load B; 
load Mr; load Jr; load Br;

sys_full = dss(J,B,B',0,M);
sys_full = minreal(sys_full);


sys_red = dss(Jr,Br,Br',0,Mr);
sys_red = minreal(sys_red);

for i=1:6
    figure(i); sigma(sys_full(i,i), 'b', sys_red(i,i), 'r')
    legend('EB full', 'EB n = 30')
end

