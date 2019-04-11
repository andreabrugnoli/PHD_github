# General PHDAE
import numpy as np


class SystemPHDAE:

    def __init__(self, J, B, E=None, Q=None, R=None):

        self.J = J
        self.B = B

        if E is not None:
            self.E = E
        else:
            self.E = np.ones_like(J)

        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.ones_like(J)

        if R is not None:
            self.R = R
        else:
            self.R = np.zeros_like(J)
