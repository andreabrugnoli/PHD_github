load B; load Br; load E; load Er; load J; load Jr;



sys_full = dss(J,B,B',0,E) ;
sys_red = dss(Jr,Br,Br',0,Er);

figure()
bode(sys_full)
hold on
bode(sys_red)

eig_full = eig(E, J)
eig_red = eig(Er, Jr)
