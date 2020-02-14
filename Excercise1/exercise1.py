import torch
from typing import Tuple


def sgd_factorise(A: torch.Tensor, rank: int, num_epochs=1000, lr=0.01) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A.shape
    U = torch.rand(m, rank)
    V = torch.rand(n, rank)
    for epoch in range(0, num_epochs):
        for r in range(0, m):
            for c in range(0, n):
                e = A[r, c] - U[r, :] @ V[c, :].t()
                U[r, :] = U[r, :] + lr * e * V[c, :]
                V[c, :] = V[c, :] + lr * e * U[r, :]
    return U, V



def truncatedSVD(A: torch.Tensor):

    U, S, V = torch.svd(A)
    m = S.shape[0] -1
    S[m] = 0
    return  U, S, V


def sgd_factorise_masked(A: torch.Tensor, M: torch.Tensor, rank: int, num_epochs=1000,
                         lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A.shape
    U = torch.rand(m, rank)
    V = torch.rand(n, rank)
    for epoch in range(0, num_epochs):
        for r in range(0, m):
            for c in range(0, n):
                if M[r, c] == 1:
                    e = A[r, c] - U[r, :] @ V[c, :].t()
                    U[r, :] = U[r, :] + lr * e * V[c, :]
                    V[c, :] = V[c, :] + lr * e * U[r, :]
    return U, V





if __name__ == '__main__':

    test = torch.tensor([[0.3374, 0.6005, 0.1735],
                         [3.3359, 0.0492, 1.8374],
                         [2.9407, 0.5301, 2.2620]])

    U, V = sgd_factorise(test, 2)
    loss = torch.nn.functional.mse_loss(U@V.t(), test, reduction='sum')
    print(f"Approximation {U@V.t()}")
    print(f'Loss  is {loss}')\

    U, S , V = truncatedSVD(test)
    reconstruction = U @ torch.diag(S) @ V.t()
    loss = torch.nn.functional.mse_loss(reconstruction, test, reduction='sum')
    print(f"Approximation \n {reconstruction}")
    print(f'Loss  is {loss}')

    test_2 = torch.tensor([[0.3374, 0.6005, 0.1735],
                         [0, 0.0492, 1.8374],
                         [2.9407, 0, 2.2620]])

    mask = torch.tensor([[1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])

    U, V = sgd_factorise_masked(test_2, mask, 2)
    loss = torch.nn.functional.mse_loss(U @ V.t(), test, reduction='sum')
    print(f"Approximation \n {U @ V.t()}")
    print(f'Loss  is {loss}')