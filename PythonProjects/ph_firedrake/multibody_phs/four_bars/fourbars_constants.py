rho = 2714.4716  # kg/m^3
E = 7.1016 * 10**10  # Pa

L_ground = 0.254  # m

L_crank = 0.107949999999  # m
A_crank = 1.0774172 * 10**(-4)

I_crank = 3.881e-4*(2.54e-2)**4
EI_crank = E*I_crank # N m^2

# EI_crank = 11.472  # N m^2

L_coupler = 0.2794  # m
A_coupler = 4.064508e-05
I_coupler = 2.084e-5*(2.54e-2)**4
EI_coupler = E*I_coupler
# EI_coupler = 0.616  # N m^2

L_follower = 0.27051  # m
A_follower = 4.064508e-05
I_follower = I_coupler
EI_follower = E *I_follower
# EI_follower = 0.616  # N m^2


# I_crank = EI_crank/E
# I_coupler = EI_coupler/E
# I_follower = EI_follower/E

m_link = 0.041942966  # kg
