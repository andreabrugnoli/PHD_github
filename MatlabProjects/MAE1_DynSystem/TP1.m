%% Ex 3

num = [2 1]
dena = [10 1]
denb = [1 1]
Fa = tf(num, dena)

Fb = tf(1, denb)

F4 = Fa*Fb


%% Ex 4

s = tf('s')
G1 = 1/(1+s)
G2 = 1/(1+10*s)
G3 = 1/s
G4 = (1+10*s)/(1+20*s)
G5 = 1/(1 + 3*s)

F1 = minreal(G3*G2*G1/(1+G3*G2*G4))
F2 = minreal(G3*G2*G5/(1 + G3*G2*G4))
F3 = minreal(G1 - G4*F1)
F4 = minreal(-G4*F2)

step(F1)
step(F2)
step(F3)
step(F4)


%% eX 5
s = tf('s')
F = (1-s)/((1 + 2*s)*(1 + 10*s))

step(F)
