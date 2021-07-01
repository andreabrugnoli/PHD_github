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

path_data = "Data_Wave/"

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


# Wandb login
import wandb
wandb.login()
wandb.init(project='optimal-energy-shaping', name='wave_r01_t4')


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
            u = self._energy_shaping(csi) + self._damping_injection(Y, csi)

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

    def _damping_injection(self, vel, pos):
        # damping injection control action
        inp_K = torch.vstack((vel, pos.reshape((1, -1)))).T

        return -torch.einsum('ij,j...->ij', vel, self.K(inp_K))

    def _autonomous_energy(self, X):
        # Hamiltonian (total energy) of the UNCONTROLLED system
        return 0.5 * torch.einsum('mj, mn, nj -> j', X, self.M, X)

    def _added_energy(self, X):
        # Hamiltonian (total energy) of the CONTROLLED system
        sig = X[dofs_sig, :]
        N_el = np.size(sig, axis=0)
        csi = 1 / N_el * torch.einsum('ij -> j', sig)

        return self.H_a(csi)

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

        self.nfe = 0

    def forward(self, t, X):
        with torch.set_grad_enabled(True):
            vel, sig = X[dofs_v, :], X[dofs_sig, :]

            N_el = np.size(sig, axis=0)

            csi = 1 / N_el * torch.einsum('ij -> j', sig).reshape((-1, 1))
            csi = csi.requires_grad_(True)

            # compute control action

            Y = torch.einsum('ij, jk -> ik', self.f.C, X)
            u_ES = self.f._energy_shaping(csi)
            u_DI = self.f._damping_injection(Y, csi)

            u = u_DI + u_ES

        return torch.abs(u)


trainloader = dummy_trainloader()

e0 = torch.Tensor(e0_np).to(device).float()
eT = torch.Tensor(eT_np).to(device).float()

Eigens = torch.Tensor(Eigens_np).to(device).float()
x_dim, u_dim = len(e0), 1

n_instances = 250


class EnergyShapingLearner(pl.LightningModule):
    def __init__(self, f: nn.Module, t_span, sensitivity='adjoint'):
        super().__init__()
        self.model = f
        self.t_span = t_span
        self.odeint = odeint if sensitivity == 'autograd' else odeint_adjoint

        # Uniform distribution for initial conditions

        self.coeff_z0 = (5 - 8 * torch.rand(x_dim, n_instances)).to(device)

        self.z0 = torch.einsum('ij, jk -> ik', Eigens, self.coeff_z0)

        self.model.nfe = 0
        self.solver = 'rk4'
        self.solver_options = dict(step_size=5e-2)
        self.nfe = [0]

        self.lr = 1e-3

    def forward(self, x0):
        zT = odeint(self.model, x0, self.t_span,
                    method=self.solver, options=self.solver_options)

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

        loss = terminal_loss + 0.1 * integral_loss

        # log training data
        self.logger.experiment.log(
            {
                'terminal loss': terminal_loss,
                'integral_loss': integral_loss,
                'train loss': loss,
                'nfe': self.model.nfe,
                'v_max': xT[dofs_v, :].max(),
                'sig_max': xT[dofs_sig, :].max(),
                'v_min': xT[dofs_v, :].min(),
                'sig_min': xT[dofs_sig, :].min(),
                'xT_mean': xT.mean(),
                'xT_std': xT.std()
            }
        )

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
h_dim = 32
H_a = nn.Sequential(
          nn.Linear(1, h_dim),
          nn.Tanh(),
          nn.Linear(h_dim, h_dim),
          nn.Tanh(),
          nn.Linear(h_dim, 1))

# damping dependent on position and velocity
K = nn.Sequential(
         nn.Linear(2, h_dim),
         nn.Sigmoid(),
         nn.Linear(h_dim, h_dim),
         nn.Sigmoid(),
         nn.Linear(h_dim, 1),
         nn.Softplus()).to(device)

for p in H_a[-1].parameters(): torch.nn.init.zeros_(p)
for p in K[-2].parameters(): torch.nn.init.zeros_(p)

f = ControlledSystem(A, B, C, M, K, H_a).to(device)
aug_f = AugmentedDynamics(f, ControlEffort(f))

n_t = 500
t_fin = 4

t_span= torch.linspace(0, t_fin, n_t).to(device)

learn = EnergyShapingLearner(aug_f, t_span)
learn.lr = 5e-3
logger = WandbLogger(project='optimal-energy-shaping', name='wave_r01_t4')

trainer = pl.Trainer(max_epochs=30, gpus=1, gradient_clip_val=1., logger=logger)
# trainer = pl.Trainer(max_epochs=10, gpus=None, gradient_clip_val=1.)

trainer.fit(learn)


# Testing the control
n_ic = 50

coeff_x0 = torch.randn(x_dim, n_ic).to(device)
x0 = torch.einsum('ij, jk -> ik', Eigens, coeff_x0)

# x0 =learn.z0

x0_aug = torch.cat([x0, torch.zeros(1, x0.shape[1]).to(device)], 0)

model = aug_f.to(device)

t_test = torch.linspace(0, 10, n_t).to(device)
traj = odeint(model, x0_aug, t_test, method='rk4',
              options=dict(step_size=5e-2)).detach().cpu()


# Plot of test
# plot

mean_traj = traj.mean(axis=2)

plt.figure()
n_in = 0 #int(4/5*n_t)
n_fin=-1
plt.plot(t_test.cpu()[n_in:n_fin], mean_traj[n_in:n_fin,dofs_v], ':k')

plt.figure()
plt.plot(t_test.cpu()[n_in:n_fin], mean_traj[n_in:n_fin,dofs_sig], ':k')



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

plt.figure()
n_grid = 100
pos_grid, vel_grid = torch.linspace(-1, 1, n_grid), torch.linspace(-2, 2, n_grid)
Wdot_L, W_L = torch.meshgrid(vel_grid, pos_grid)
# z = torch.cat([Wdot_L.reshape(-1, 1), W_L.reshape(-1, 1)], 1)

# print(W_L.is_cuda)

# print(Wdot_L.is_cuda)

Kd_grid = -model.f._damping_injection(Wdot_L.reshape(1, -1).cuda(), W_L.reshape(-1, 1).cuda()).\
 reshape(n_grid, n_grid).detach().cpu()/Wdot_L
plot = plt.contourf(Wdot_L, W_L, Kd_grid, 100, cmap='inferno')
plt.colorbar(plot)
# y_vec = torch.linspace(-2, 2, 200).reshape((-1, 1)).to(device).float()

# K_vec = model.f.K(y_vec)

# plt.plot(y_vec.detach().cpu(), K_vec.detach().cpu(), 'b')

plt.figure()
csi_vec = torch.linspace(-1, 1, 200).reshape((-1, 1)).to(device).float()

Ha_vec = model.f.H_a(csi_vec)

plt.plot(csi_vec.detach().cpu(), Ha_vec.detach().cpu(), 'b')

u_es_sec = model.f._energy_shaping(csi_vec.requires_grad_())

print(u_es_sec[0,-1])
plt.figure()
plt.plot(csi_vec.detach().cpu(), u_es_sec.T.detach().cpu(), 'b')


