import torch

X = torch.tensor([[1, 2, 3],[4, 5, 6], [7, 8, 9]])
K = torch.tensor([[1, 2, 3]]).T

KX = torch.einsum('ij,j...->ij', X, K)

print(KX)