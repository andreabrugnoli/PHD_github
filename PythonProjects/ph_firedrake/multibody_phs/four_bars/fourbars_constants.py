rho = 2714  # kg/m^3
E = 7.1016 * 10**10  # Pa

L_ground = 0.254  # m

L_crank = 0.108  # m
A_crank = 1.0774 * 10**(-4)
EI_crank = 11.472  # N m^2

L_coupler = 0.2794  # m
A_coupler = 4.0645 * 10**(-5)
EI_coupler = 0.616  # N m^2

L_follower = 0.2705  # m
A_follower = 4.0645 * 10**(-5)
EI_follower = 0.616  # N m^2

I_crank = EI_crank/E
I_coupler = EI_coupler/E
I_follower = EI_follower/E

m_link = 0.042  # kg
