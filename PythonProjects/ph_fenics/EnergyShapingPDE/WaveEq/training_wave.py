import torch
from torch import nn
import torch.utils.data as data
from torch.autograd import grad as grad

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from numpy import pi

import numpy as np
from torchdiffeq import odeint, odeint_adjoint
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt

path_data = "/home/andrea/PHD_github/PythonProjects/ph_fenics/EnergyShapingPDE/WaveEq/Data_Wave/"

e0_np = np.load(path_data + 'e0.npy').reshape((-1,1))
eT_np = np.load(path_data + 'eT.npy').reshape((-1,1))

M_np = np.load(path_data + 'M.npy')
J_np = np.load(path_data + 'J.npy')
B_np = np.load(path_data + 'B.npy')


A_sys = np.linalg.solve(M_np, J_np)
B_sys = np.linalg.solve(M_np, B_np).reshape((-1, 1))
C_sys = B_np.reshape((1, -1))


dofs_dict = np.load(path_data + 'dofs_dict.npy', allow_pickle=True)
x_dict = np.load(path_data + 'x_dict.npy', allow_pickle=True)

Eigens_np = np.load(path_data + 'eigvectors.npy')

dofs_v = dofs_dict.item().get('v')
dofs_sig = dofs_dict.item().get('sig')

x_v = x_dict.item().get('v')
x_sig = x_dict.item().get('sig')


def dummy_trainloader():
    tl = data.DataLoader(data.TensorDataset(torch.Tensor(1), torch.Tensor(1)), batch_size=1, num_workers=8)
    return tl


class ControlledSystem(nn.Module):
    def __init__(self, A, B, C, M, K, H_a):
        super().__init__()
        self.A = A
        self.B = B
        self.C = C
        self.M = M

        self.K = K
        self.H_a = H_a

        self.nfe = 0

    def forward(self, t, X):
        with torch.set_grad_enabled(True):
            vel, sig = X[dofs_v, :], X[dofs_sig, :]
            N_el = np.size(sig, axis=0)

            csi = 1 / N_el * torch.einsum('ij -> j', sig).reshape((-1, 1))
            csi = csi.requires_grad_(True)

            # compute control action
            Y = torch.einsum('ij, jk -> ik', self.C, X)
            u = self._energy_shaping(csi) + self._damping_injection(Y)

            # compute dynamics
            dXdt = self._dynamics(X, u)

            self.nfe += 1

        return dXdt

    def _dynamics(self, X, u):
        dX = torch.einsum('ij,jk->ik', self.A, X) + torch.einsum('ij, jk -> ik', self.B, u)

        return dX

    def _energy_shaping(self, csi):
        # energy shaping control action
        dH_a = grad(self.H_a(csi).sum(), csi, create_graph=True)[0]
        return -dH_a.T

    def _damping_injection(self, Y):
        # damping injection control action

        return -torch.einsum('ij,j...->ij', Y, self.K(Y.T))

    def _autonomous_energy(self, X):
        # Hamiltonian (total energy) of the UNCONTROLLED system
        return 0.5 * torch.einsum('mj, mn, nj -> j', X, self.M, X)

    def _energy(self, X):
        # Hamiltonian (total energy) of the CONTROLLED system
        sig = X[dofs_sig, :]
        N_el = np.size(sig, axis=0)
        csi = 1 / N_el * torch.einsum('ij -> j', sig)

        return self._autonomous_energy(X) + self.H_a(csi)


class AugmentedDynamics(nn.Module):
    # "augmented" vector field to take into account integral loss functions
    def __init__(self, f, int_loss):
        super().__init__()
        self.f = f
        self.int_loss = int_loss
        self.nfe = 0.

    def forward(self, t, X):
        self.nfe += 1
        X = X[:-1, :]
        dXdt = self.f(t, X)
        dldt = self.int_loss(t, X)

        self.nfe += 1

        return torch.cat([dXdt, dldt], 0)


class ControlEffort(nn.Module):
    # control effort integral cost
    def __init__(self, f):
        super().__init__()
        self.f = f

        self.nfe= 0

    def forward(self, t, X):
        with torch.set_grad_enabled(True):
            vel, sig = X[dofs_v, :], X[dofs_sig, :]

            N_el = np.size(sig, axis=0)

            csi = 1 / N_el * torch.einsum('ij -> j', sig).reshape((-1, 1))
            csi = csi.requires_grad_(True)

            # compute control action

            Y = torch.einsum('ij, jk -> ik', self.f.C, X)
            u_ES = self.f._energy_shaping(csi)
            u_DI = self.f._damping_injection(Y)

            u = u_DI + u_ES

        return torch.abs(u)


trainloader = dummy_trainloader()

e0 = torch.Tensor(e0_np).to(device).float()
eT = torch.Tensor(eT_np).to(device).float()

Eigens = torch.Tensor(Eigens_np).to(device).float()
x_dim, u_dim = len(e0), 1

n_instances = 30

class EnergyShapingLearner(pl.LightningModule):
    def __init__(self, f: nn.Module, t_span, sensitivity='autograd'):
        super().__init__()
        self.model = f
        self.t_span = t_span
        self.odeint = odeint if sensitivity == 'autograd' else odeint_adjoint

        self.coeff_z0 = torch.randn(x_dim, n_instances).to(device)

        self.z0 = torch.einsum('ij, jk -> ik', Eigens, self.coeff_z0)

        self.model.nfe = 0
        self.solver = 'dopri5'
        self.nfe = [0]

        self.lr = 1e-3

    def forward(self, x0):

        zT = odeint(self.model, x0, self.t_span,
                    method=self.solver, rtol=1e-4, atol=1e-4)

        return zT

    def training_step(self, batch, batch_idx):

        self.model.nfe = 0
        x0 = torch.cat([self.z0, torch.zeros(1, n_instances).to(self.z0)], 0)
        xTl = self(x0)


        xT, l = xTl[:, :-1, :], xTl[:, -1, :]

        self.nfe += [self.model.nfe]
        # print(self.nfe)

        # Compute loss
        terminal_loss = torch.norm(xT[-1, :, :] - eT, p=2, dim=0).mean()

        integral_loss = torch.mean(l)

        loss = terminal_loss + 0.01 * integral_loss

        # log training data
        # self.logger.experiment.log(
        #     {
        #         'terminal loss': terminal_loss,
        #         'integral_loss': integral_loss,
        #         'train loss': loss,
        #         'nfe': self.model.nfe,
        #         'v_max': xT[dofs_v, :].max(),
        #         'sig_max': xT[dofs_sig, :].max(),
        #         'v_min': xT[dofs_v, :].min(),
        #         'sig_min': xT[dofs_sig, :].min(),
        #         'xT_mean': xT.mean(),
        #         'xT_std': xT.std()
        #     }
        # )

        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return trainloader


A = torch.Tensor(A_sys).to(device).float()
B = torch.Tensor(B_sys).to(device).float()
C = torch.Tensor(C_sys).to(device).float()
M = torch.Tensor(M_np).to(device).float()

# vector field parametrized by a NN
h_dim = 64
H_a = nn.Sequential(
          nn.Linear(1, h_dim),
          nn.Softplus(),
          nn.Linear(h_dim, h_dim),
          nn.Tanh(),
          nn.Linear(h_dim, h_dim),
          nn.Tanh(),
          nn.Linear(h_dim, 1))

K = nn.Sequential(
         nn.Linear(1, h_dim),
         nn.Sigmoid(),
         nn.Linear(h_dim, h_dim),
         nn.Sigmoid(),
         nn.Linear(h_dim, 1),
         nn.Softplus()).to(device)

for p in H_a[-1].parameters(): torch.nn.init.zeros_(p)
for p in K[-2].parameters(): torch.nn.init.zeros_(p)

f = ControlledSystem(A, B, C, M, K, H_a).to(device)
aug_f = AugmentedDynamics(f, ControlEffort(f))

n_t = 100
t_fin = 3

t_span= torch.linspace(0, t_fin, n_t).to(device)

learn = EnergyShapingLearner(aug_f, t_span)
learn.lr = 5e-3
logger = WandbLogger(project='optimal-energy-shaping', name='wave')

# trainer = pl.Trainer(max_epochs=100, gpus=1, gradient_clip_val=1., logger=logger)
trainer = pl.Trainer(max_epochs=10, gpus=None, gradient_clip_val=1.)

trainer.fit(learn)


# xT = odeint(learn.model.to(device), learn.z0.to(device), learn.t, method='dopri5').detach().cpu()
#
# xT_1 = odeint(learn.model.to(device), x0, learn.t, method='dopri5').detach().cpu()
#
# fig, axs = plt.subplots(2, 1)
# for i in range(n_instances):
#     axs[0].plot(learn.t_span.cpu(), xT[:,:,i], ':k')
# axs[1].plot(learn.t.cpu(), xT_1[:,:,0], ':k')

# uT = f.u(xT.to(device))

# fig, axs = plt.subplots(1, 1, figsize=(10, 5))
# axs[0].plot(t.cpu(), xT[:,0,:], ':k');
# # axs[1].plot(t.cpu(), uT[:,0,:].detach().cpu(), ':b');

# sol_torch = torch.Tensor(xT).to(device).float()
# H_vec = 0.5*torch.einsum('ijk, jm, imk -> ik', sol_torch, M,  sol_torch)
#
# meanH = torch.mean(H_vec, dim=1)
#
# sol_torch_1 = torch.Tensor(xT_1).to(device).float()
# H_vec_1 = 0.5*torch.einsum('ijk, jm, imk -> i', sol_torch_1, M,  sol_torch_1)
#
# plt.figure()
# plt.plot(learn_dopri.t.cpu(), meanH.cpu(), 'b-')
# plt.plot(learn_dopri.t.cpu(), H_vec_1.cpu(), 'r-')

# y_vec = np.linspace(-5, 5, 200).reshape((-1, 1))
#
# K_vec = K(torch.Tensor(y_vec).to(device).float())
#
# plt.plot(y_vec, K_vec.detach().cpu(), 'b')

# vT = xT[:,dofs_v, 0]
# sigT = xT[:,dofs_sig, 0]
#
#
# fig, axs = plt.subplots(2, 1)
# axs[0].scatter(x_v, vT[0], color='b')
# axs[1].scatter(x_v, vT[-1], color='r')

# axs[1].scatter(x_sig, sigT[0], color='r')