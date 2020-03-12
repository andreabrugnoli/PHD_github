function dtalpha = HeatNNpHs(t, alpha, D, Bnd, L, uL, uR, Nx)
    dtalpha(1:Nx) = D * alpha(Nx+1:2*Nx) + Bnd * [uL(t); uR(t)];
    dtalpha(Nx+1:2*Nx) = D' * alpha(1:Nx) + L * alpha(Nx+1:2*Nx); % With Lambda^{-1}
    dtalpha = dtalpha';
end

