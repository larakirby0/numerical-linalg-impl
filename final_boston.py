import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

# constants
from constants import epsilon


def load_boston():
    """The deprecated boston dataset."""
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)

    n_rows = raw_df.shape[0] // 2
    data = np.zeros((n_rows, 13))
    target = np.zeros(n_rows)

    for i in range(n_rows):
        row1 = raw_df.iloc[2 * i, :].to_numpy()
        row2 = raw_df.iloc[2 * i + 1, :].to_numpy()
        data[i, :] = np.hstack([row1, row2[:2]])
        target[i] = row2[2]

    feature_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df["PRICE"] = target
    return df


def report_time(func):
    """Print how long calling the function took.

    Args:
        func (Callable): The function we're calling.
    """

    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        runtime = time() - start
        print(f"Running {func.__name__} took {runtime} seconds.")
        return result

    return wrapper


def power_method_single(A: np.array) -> tuple:
    """Get the first eigenvalue by using the power method.

    Args:
        A (np.array): A matrix.

    Returns:
        tuple: (eigenvalue, associated eigenvector)
    """
    x_k = np.random.rand(A.shape[0])
    x_k = x_k / np.linalg.norm(x_k)
    lambda_k = 100
    error = 1

    while error >= epsilon:
        x_k_1 = np.dot(A, x_k)
        new_x = x_k_1 / np.linalg.norm(x_k_1)
        new_lambda = (new_x.T @ A @ new_x) / (new_x.T @ new_x)
        lambda_k = new_lambda
        x_k = new_x

        error = np.linalg.norm(A @ new_x - (lambda_k * x_k))
    return float(lambda_k), x_k


@report_time
def power_method_with_deflation(A: np.array, k=None) -> list:
    """Determine the eigenpairs for a matrix A using the power method with deflation.
    We find the last eigenvalue with the power method, then compute a new A which has
    the next eigenvalue as its dominant one.

    Args:
        A (np.array): a matrix.
        k (Optional[int]): limit amount of eigenpairs. Defaults to no limit/all.

    Returns:
        list: [(lambda_i, x_i): i <=k] eigenpairs
    """
    # user has the option to input an amount of eigenvalues they want, but we will get all eigenpairs
    k = k or A.shape[0]

    lambda_1, v_1 = power_method_single(A)
    eigenpairs = [(lambda_1, v_1)]

    current_A = A.copy().astype(float)
    current_lambda = lambda_1
    current_v = v_1
    for _ in range(2, k + 1):
        # create a new Ak such that the next eigenvalue is dominant
        A_k = current_A - current_lambda * np.outer(current_v, current_v)
        # retrieve the eigenpair
        lambda_k, v_k = power_method_single(A_k)
        # add to list- we can be sure these are in descending order since we find the largest first
        eigenpairs.append((lambda_k, v_k))
        current_v = v_k
        current_lambda = lambda_k
        current_A = A_k

    return eigenpairs


def hessenberg_inverse_variation(l, H, P) -> np.array:
    """Return approximate eigenvector given an eigenvalue.
    Algorithm 9.8.

    Args:
        l (float): lambda, the eigenvalue
        H (np.array): The transformed Hessenberg matrix.
        P (np.array): Our orthogonal similarity transformer.

    Returns:
        np.array: the eigenvector associated with lambda.
    """
    error = 1
    # initialize a random guess
    y = np.random.rand(H.shape[0])
    y /= np.linalg.norm(y)
    max_iter = 1000
    i = 0

    while error >= epsilon and i < max_iter:
        z_k = np.linalg.solve(H - l * np.eye(H.shape[0]), y)
        if not np.linalg.norm(z_k):
            # if the norm of z_k is zero, try another guess.
            y = np.random.rand(H.shape[0])
            y /= np.linalg.norm(y)
            continue
        z_k /= np.linalg.norm(z_k)
        error = np.linalg.norm(H @ z_k - l * z_k)

        # check for convergence
        if error < epsilon:
            return P @ z_k
        y = z_k
        i += 1

    return P @ z_k


def get_householder(vector) -> tuple:
    """Get a householder vector u and scalar x such that H_u @ vector = (x, 0, 0...).
    This is specific to the case where we want one value, then zeros, not a general HH vector solver.

    Args:
        vector (np.array): input vector.

    Returns:
        tuple: (u, x): householder vector and scalar.
    """
    scaling_factor = np.linalg.norm(vector, ord=np.inf)
    scaled_vector = vector / scaling_factor
    u = scaled_vector.copy()

    u_norm = np.linalg.norm(u, ord=2)
    sign = 1 if scaled_vector[0] >= 0 else -1
    u[0] = u[0] + sign * u_norm
    return u, scaling_factor


def hessenberg_transformation(A: np.array) -> tuple:
    """Determine orthogonal P such that PAPt is Hessenberg using Algorithm 9.6.

    Args:
        A (np.array): a matrix.

    Returns:
        tuple[np.array]: Hessenberg version of A, Orthogonal transformation matrix P.
    """
    A_copy = A.astype(float).copy()
    n = A.shape[0]
    P = np.eye(n)

    for k in range(n - 2):
        column = A_copy[k + 1 :, k]
        column_norm = np.linalg.norm(column)
        sign = 1 if column[0] >= 0 else -1

        zeros_n_1 = np.zeros(len(column))
        zeros_n_1[0] = 1
        uk = column + sign * column_norm * zeros_n_1
        uk = uk / np.linalg.norm(uk)

        # update A to transform it
        A_copy[k + 1 :, k:] -= 2 * np.outer(uk, uk @ A_copy[k + 1 :, k:])
        A_copy[:, k + 1 :] -= 2 * np.outer(A_copy[:, k + 1 :] @ uk, uk)

        # update the P we use to transform A on either side
        P[:, k + 1 :] -= 2 * np.outer(P[:, k + 1 :] @ uk, uk)
    return A_copy, P


@report_time
def basic_qr_iteration(A) -> list:
    """Determine eigenpairs via basic QR iteration after transforming A via householder transformations.
    Algorithm 9.6.

        Args:
            A (np.array): input matrix.

        Returns:
            List(tuple): list of eigenvalue/eigenvector pairs.
    """
    # 1: perform orthogonal similarity transformations to make a hessenberg (tridiagonal if symmetric)
    H, P = hessenberg_transformation(A)

    A_k_minus_1 = H.copy()
    error = 1
    while error >= epsilon:

        # 2: QR factorization
        Q_k, R_k = np.linalg.qr(A_k_minus_1)

        # 3: R_k @ Q_k will be orthogonally similar to the original matrix
        A_k = R_k @ Q_k

        # check for convergence
        error = np.linalg.norm(A_k - A_k_minus_1, ord="fro")
        A_k_minus_1 = A_k

    # 4: R_n @ Q_n has the same(ish) eigenvalues as A since all were orthogonally similar.
    eigenvalues = np.diag(A_k_minus_1).astype(float)
    eigenvectors = [hessenberg_inverse_variation(l, H, P) for l in eigenvalues]

    # ensure the eigenpairs are returned associated correctly in descending order
    zipped_pairs = list(zip(eigenvalues, eigenvectors))
    zipped_pairs.sort(key=lambda x: x[0], reverse=True)

    return zipped_pairs


def wilkinson_shift(T) -> float:
    """Find the wilkinson shift for a matrix T.

    Args:
        T (np.array): input matrix

    Returns:
        float: Wilkinson Shift
    """
    row = T.shape[0] - 1

    while row > 0 and T[row, row - 1] == 0:
        row -= 1

    if row == 0:
        return T[0, 0]

    r = (T[row - 1, row - 1] - T[row, row]) / 2

    return T[row, row] + r - (np.sign(r) or 1) * np.sqrt(r**2 + T[row, row - 1] ** 2)


@report_time
def symmetric_qr_iteration_with_shift(A) -> list:
    """Determine eigenpairs via symmetric QR iteration with Wilkinson Shift.
    Algorithm 10.2.

        Args:
            A (np.array): input matrix, assumed to be symmetric.

        Returns:
            List(tuple): list of eigenvalue/eigenvector pairs.
    """
    # 1: perform orthogonal similarity transformations to make a tridiagonal matrix
    T, P = hessenberg_transformation(A)

    Tk_minus_one = T.copy().astype(float)
    n = T.shape[0]
    error = 1
    while error >= epsilon:
        # 2: calculate Wilkinson Shift
        mu = wilkinson_shift(Tk_minus_one)

        # 3: QR factorization
        Qk, Rk = np.linalg.qr(Tk_minus_one - mu * np.eye(n))
        Tk = Rk @ Qk + mu * np.eye(n)

        # 4: check for convergence
        error = np.linalg.norm(Tk - Tk_minus_one, ord="fro")
        Tk_minus_one = Tk

    eigenvalues = np.diag(Tk_minus_one).astype(float)
    eigenvectors = [hessenberg_inverse_variation(l, T, P) for l in eigenvalues]

    # ensure the eigenpairs are returned associated correctly in descending order
    zipped_pairs = list(zip(eigenvalues, eigenvectors))
    zipped_pairs.sort(key=lambda x: x[0], reverse=True)

    return zipped_pairs


def eigenval_residual(A, eigenpairs) -> float:
    """Get Eigenvalue residuals for A.

    Args:
        A (np.array): A matrix.
        eigenpairs (list(tuple)): a list [(lambda, x_k)]

    Returns:
        float: the Eigenvalue residual.
    """
    return max([np.linalg.norm(A @ e - l * e) for (l, e) in eigenpairs])


def get_cumulative_variance(eigpairs):
    # Cumulative variance measure as defined in the assignment spec
    eigvals = np.array([l[0] for l in eigpairs])
    explained_variance = eigvals / eigvals.sum()
    return np.cumsum(explained_variance)


def get_reconstruction_error(data, eigpairs):
    # Reconstruction error as defined in the assignment spec
    sorted_eigpairs = sorted(eigpairs, reverse=True)
    vecs = np.column_stack([v for (_, v) in sorted_eigpairs])

    m = data.shape[1]
    errors = []

    for i in range(1, m + 1):
        W_k = vecs[:, :i]
        Z = (data @ W_k) @ W_k.T
        error = np.linalg.norm(data - Z, "fro")
        errors.append(error)
    return errors


# CALLING AND TESTING
A = [
    [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [9, 9, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1],
    [7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1],
    [6, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2, 1],
    [5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
A = np.array(A)

print("Running methods on A.")

deflation = power_method_with_deflation(A, A.shape[0])
basic_qr = basic_qr_iteration(A)
shift_qr = symmetric_qr_iteration_with_shift(A)

deflation_resid = eigenval_residual(A, deflation)
basic_qr_resid = eigenval_residual(A, basic_qr)
shift_qr_resid = eigenval_residual(A, shift_qr)

print("Deflation eigenvalue residual: A", deflation_resid)
print("Basic QR eigenvalue residual: A", basic_qr_resid)
print("Shift QR eigenvalue residual: A", deflation_resid)


def boston_pca():
    boston = load_boston()

    # standardize
    means = boston.mean()
    stdevs = boston.std()
    boston = (boston - means) / stdevs
    boston_array = boston.to_numpy()
    C = boston_array.T @ boston_array / (boston_array.shape[0] - 1)

    # find eigenpairs via several methods
    deflation_eigenpairs = power_method_with_deflation(C)
    qr_eigenpairs = basic_qr_iteration(C)
    shift_qr_eigenpairs = symmetric_qr_iteration_with_shift(C)

    qr_cumulative_variance = get_cumulative_variance(qr_eigenpairs)
    shift_cumulative_variance = get_cumulative_variance(shift_qr_eigenpairs)
    deflation_cumulative_variance = get_cumulative_variance(deflation_eigenpairs)

    # plot  QR explained variance and reconstruction error
    qr_cumulative_variance = get_cumulative_variance(qr_eigenpairs)
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(qr_cumulative_variance) + 1), qr_cumulative_variance, marker="o"
    )

    plt.xlabel("components")
    plt.ylabel("cumulative explained variance")
    plt.title("QR method explained variance")
    plt.grid(True)
    plt.show(block=False)

    plt.figure(figsize=(8, 5))
    qr_errors = get_reconstruction_error(C, qr_eigenpairs)
    plt.plot(range(1, len(qr_eigenpairs) + 1), qr_errors, marker="o")
    plt.xlabel("components")
    plt.ylabel("reconstruction error")
    plt.title("QR reconstruction error")
    plt.grid(True)
    plt.show(block=False)

    # plot shift QR explained variance and reconstruction error
    shift_cumulative_variance = get_cumulative_variance(shift_qr_eigenpairs)
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(shift_cumulative_variance) + 1),
        shift_cumulative_variance,
        marker="o",
    )

    plt.xlabel("components")
    plt.ylabel("cumulative explained variance")
    plt.title("Shift QR method explained variance")
    plt.grid(True)
    plt.show(block=False)

    plt.figure(figsize=(8, 5))
    shift_errors = get_reconstruction_error(C, shift_qr_eigenpairs)
    plt.plot(range(1, len(shift_qr_eigenpairs) + 1), shift_errors, marker="o")
    plt.xlabel("components")
    plt.ylabel("reconstruction error")
    plt.title("Shift QR reconstruction error")
    plt.grid(True)
    plt.show(block=False)

    # plot Power Method explained variance and reconstruction error
    deflation_cumulative_variance = get_cumulative_variance(deflation_eigenpairs)
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(deflation_cumulative_variance) + 1),
        deflation_cumulative_variance,
        marker="o",
    )

    plt.xlabel("components")
    plt.ylabel("cumulative explained variance")
    plt.title("Power method explained variance")
    plt.grid(True)
    plt.show(block=False)

    plt.figure(figsize=(8, 5))
    deflation_errors = get_reconstruction_error(C, deflation_eigenpairs)
    plt.plot(range(1, len(deflation_eigenpairs) + 1), deflation_errors, marker="o")
    plt.xlabel("components")
    plt.ylabel("reconstruction error")
    plt.title("Power method reconstruction error")
    plt.grid(True)
    plt.show()

    print("Power method residuals:", eigenval_residual(C, deflation_eigenpairs))
    print("Basic QR residuals:", eigenval_residual(C, qr_eigenpairs))
    print("Shifted QR residuals:", eigenval_residual(C, shift_qr_eigenpairs))


print("Running PCA Analysis.")
boston_pca()
