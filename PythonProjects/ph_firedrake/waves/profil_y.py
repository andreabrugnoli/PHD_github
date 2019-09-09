import numpy as np
import matplotlib.pyplot as plt

R = 1
r = np.linspace(0, R, 1000)
A = np.array([[R**2, R**3, R**4],
              [2*R, 3*R**2, 4*R**3],
              [(R/2)**2, (R/2)**3, (R/2)**4]])

b = np.array([0, 0, 1])

a2, a3, a4 = np.linalg.solve(A, b)
print(a2, a3, a4)

# uy = a2 * r**2 + a3*r**3 + a4*r**4
uy = 16*r**2*(1-r)**2


plt.plot(r, uy); plt.show()

