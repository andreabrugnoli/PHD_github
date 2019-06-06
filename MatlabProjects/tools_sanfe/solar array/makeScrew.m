function Mdyn = makeScrew(Mdyn,M,J,n_free)

% Screw Adding
    m = 1/1000.*blkdiag(M,J.*eye(2));
    
    %Screw on P
    Mdyn.d(3*n_free+1:3*(n_free+1),3*n_free+1:3*(n_free+1)) = Mdyn.d(3*n_free+1:3*(n_free+1),3*n_free+1:3*(n_free+1)) - m;
    
    %Screw on each C
    m_sup = [];
    for i = 1:n_free
        m_sup = blkdiag(m_sup,m);
    end
    Mdyn = feedback(Mdyn,m_sup,1:3*n_free,1:3*n_free);
    
end
