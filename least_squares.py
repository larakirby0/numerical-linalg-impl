import numpy as np


def fwd_substitution(H, c):
    n = H.shape[0]
    y = np.zeros_like(c, dtype=np.double)

    y[0] = c[0] / H[0, 0]

    for i in range(1, n):
        sum_term = np.dot(H[i, :i], y[:i])
        y[i] = (c[i] - sum_term) / H[i, i]
    return y


def bkwd_substitution(H, y):
    n = H.shape[0]
    x = np.zeros_like(y, dtype=np.double)

    for i in reversed(range(n)):
        sum_terms = np.dot(H[i, i + 1 : n], x[i + 1 : n])
        x[i] = (y[i] - sum_terms) / H[i, i]

    return x


def least_squares_normal(A, b) -> tuple:
    """Algorithm 8.1"""
    A_t = np.linalg.matrix_transpose(A)
    c = np.dot(A_t, b)

    A_t_A = np.dot(A_t, A)
    lower_cholesky = np.linalg.cholesky(A_t_A)
    upper_cholesky = np.transpose(lower_cholesky)

    y = fwd_substitution(lower_cholesky, c)
    x = bkwd_substitution(upper_cholesky, y)

    return x


def least_squares_qr(A, b):
    q, r = np.linalg.qr(A)  # reduced qr is default
    c = np.dot(np.linalg.matrix_transpose(q), b)
    x = np.linalg.solve(r, c)
    return x


def least_squares_svd(A, b):
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False, compute_uv=True)  # reduced SVD

    Sigma_inv = np.diag(1 / Sigma)
    b_prime = np.dot(np.transpose(U), b)
    y = Sigma_inv @ b_prime
    sol = np.dot(np.transpose(Vt), y)
    return sol


def get_error(x, x_hat):
    return np.linalg.norm(x - x_hat, ord=2) / np.linalg.norm(x, ord=2)


A = [
    [1, 1, 1, 1],
    [1, 2, 4, 8],
    [1, 3, 9, 27],
    [1, 4, 16, 64],
    [1, 5, 25, 125],
    [1, 6, 36, 36 * 6],
    [1, 7, 49, 49 * 7],
    [1, 8, 64, 64 * 8],
    [1, 9, 81, 9 * 81],
    [1, 10, 100, 1000],
]
b = np.array([np.sum(row) for row in A], dtype=np.double).reshape(-1, 1)
A = np.array(A, dtype=np.double)

b_p = np.array([np.sum(row) for row in A], dtype=np.double).reshape(-1, 1)
b_p[0] = 0

x_nem = least_squares_normal(A, b)
x_hat_nem = least_squares_normal(A, b_p)
x_nem_err = get_error(x_nem, x_hat_nem)

x_qr = least_squares_qr(A, b)
x_hat_qr = least_squares_qr(A, b_p)
x_qr_err = get_error(x_qr, x_hat_qr)

x_svd = least_squares_svd(A, b)
x_hat_svd = least_squares_svd(A, b_p)
x_svd_err = get_error(x_svd, x_hat_svd)


print("the system")
print(A, "\n", b, "\n", b_p)

print("LEAST SQUARES QR\n")
print(x_qr)
print("perturbed")
print(x_hat_qr)
print("error")
print(x_qr_err)

print("LEAST SQUARES NORMAL\n")
print(x_nem)
print("perturbed")
print(x_hat_nem)
print("error")
print(x_nem_err)

print("LEAST SQUARES SVD\n")
print(x_svd)
print("perturbed")
print(x_hat_svd)
print("error")
print(x_svd_err)
