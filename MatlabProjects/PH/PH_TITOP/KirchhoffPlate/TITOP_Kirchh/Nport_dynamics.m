function[a,b,c,d] = Nport_dynamics(nb_mode, LS, omega2, EV_C, zeta, DPa0, tauCP, n_free)

    D = 2*zeta.*sqrt(omega2); 
    K = omega2;
    Phi_b = zeros(nb_mode,n_free*3);
    Phi_c = zeros(n_free*3,nb_mode*2);
    Phi_d = zeros(n_free*3,n_free*3);
    tau_d = zeros(n_free*3,3);
    
    for i = 1:n_free
        Phi_b(:,((i-1)*3+1):3*i) = EV_C(:,:,i)';
        Phi_c( ((i-1)*3+1):3*i,:) = [-EV_C(:,:,i)*K, -EV_C(:,:,i)*D];
        for j = 1:n_free
            Phi_d((i-1)*3+1:i*3,3*(j-1)+1:3*j) = EV_C(:,:,i)*EV_C(:,:,j)';
        end
        tau_d( ((i-1)*3+1):3*i,:) = tauCP(:,:,i)-EV_C(:,:,i)*LS;
    end
    
    a = [zeros(nb_mode) eye(nb_mode);-K -D]; 
    b = [zeros(nb_mode,3*n_free) zeros(nb_mode,3); Phi_b -LS];
    c = [Phi_c; LS'*K LS'*D];
    d = [Phi_d tau_d; tau_d' -DPa0];

end