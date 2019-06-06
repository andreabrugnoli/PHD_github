load('Mdyn_SA1.mat')

addpath('./Matrices/')
load J_pH; load Q_pH; load B_pH

sys_phode = ss(J_pH*Q_pH, B_pH, B_pH'*Q_pH, 0);
sys_titop =invio(Mdyn_SA1, [1,2,3]) * tf(1, [1, 0]) ;

% figure(); sigma(Mdyn_SA1, 'r')

figure(); sigma( sys_phode, 'r', sys_titop, 'b', {1e-5, 1e6});
legend('PHODE', 'TITOP');
