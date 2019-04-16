clear all; close all; clc;
% manipulator_ph_pivot
manipulator_titop
figure(); sigma(sys_phode, 'r', sys_titop_nopivot, 'b', {w0, wf})
figure(); sigma(sys_phdae, 'r', {w0, wf}) 
figure(); sigma(sys_titop_nopivot, 'b', {w0, wf}) 