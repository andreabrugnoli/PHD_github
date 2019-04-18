function[a,b,c,d] = Nport_dynamics_Cxy(nb_mode, Lp, omega2, EV_C, Phi, zeta, DPa0, tauCP, tauCN, U_C, n_free)

    D = 2*zeta.*sqrt(omega2); 
    K = omega2;
    Phi_b = zeros(nb_mode,n_free*3);
    Phi_c = zeros(n_free*3,nb_mode*2);
    Phi_d = zeros(n_free*3,n_free*3);
    tau_d_C = zeros(3,n_free*3);
    tau_d_P = zeros(n_free*3,3);
    
    for i = 1:n_free
        Phi_b(:,((i-1)*3+1):3*i) = EV_C(:,:,i)'*tauCN(:,:,i);
        Phi_c( ((i-1)*3+1):3*i,:) = [-U_C(:,:,i)*Phi*K, -U_C(:,:,i)*Phi*D];
        for j = 1:n_free
            Phi_d((i-1)*3+1:i*3,3*(j-1)+1:3*j) = U_C(:,:,i)*Phi*EV_C(:,:,j)'*tauCN(:,:,j)';
        end
        tau_d_C(:,((i-1)*3+1):3*i) = tauCP(:,:,i)'-Lp'*EV_C(:,:,i)'*tauCN(:,:,i)';
        tau_d_P(((i-1)*3+1):3*i,:) = tauCP(:,:,i)-U_C(:,:,i)*Phi*Lp;
    end
    
    a = [zeros(nb_mode) eye(nb_mode);-K -D]; 
    b = [zeros(nb_mode,3*n_free) zeros(nb_mode,3); Phi_b -Lp];
    c = [Phi_c; Lp'*K Lp'*D];
    d = [Phi_d, tau_d_P; tau_d_C, -DPa0];

end