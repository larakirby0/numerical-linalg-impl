from math import sqrt


# SHARED FUNCTIONS
def sparse_matrix_vector_multiply(matrix_elements, row_indices, column_indices, vector):
    multiplied_vector = [0] * len(vector)
    for i in range(len(row_indices)):
        # undo the matlab 1-indexing going on
        row_index = row_indices[i] - 1
        j = column_indices[i] - 1
        multiplied_vector[row_index] += matrix_elements[i] * vector[j]
    return multiplied_vector


def diag_sparse_matrix_multiply(
    matrix_elements, row_indices, column_indices, diagonal_elements, diagonal_indices
):
    result = []
    result_row_indices = []
    result_column_indices = []
    for i in range(len(row_indices)):
        result_row_indices.append(row_indices[i])
        result_column_indices.append(column_indices[i])
        diagonal_multiplier = diagonal_elements[row_indices[i] - 1]
        result.append(matrix_elements[i] * diagonal_multiplier)
    return result, result_row_indices, result_column_indices


def subtract_vectors(a, b):
    """Subtract vector b from vector a."""
    return [(e - f) for e, f in zip(a, b)]


def add_vectors(a, b):
    """Add elements of a and b."""
    return [(e + f) for e, f in zip(a, b)]


def two_norm_squared(v):
    """Compute (||v||_{2})^2 for a vector v."""
    return sum([e**2 for e in v])


def dot_product(a, b):
    """Returns a • b"""
    return sum([e * f for e, f in zip(a, b)])


def scalar_multiply_vector(a: list, c: float):
    return [c * e for e in a]


def conjugate_gradient(ca, ci, cj, b, x_0, epsilon, max_iterations=1000):
    """Classical Conjugate Gradient algorithm for sparse matrices."""
    current_p = subtract_vectors(b, sparse_matrix_vector_multiply(ca, ci, cj, x_0))
    current_r = subtract_vectors(b, sparse_matrix_vector_multiply(ca, ci, cj, x_0))
    current_x = x_0
    # step length
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        w = sparse_matrix_vector_multiply(ca, ci, cj, current_p)
        piT_w = dot_product(current_p, w)
        alpha = two_norm_squared(current_r) / piT_w
        next_x = add_vectors(current_x, scalar_multiply_vector(current_p, alpha))
        next_r = subtract_vectors(current_r, scalar_multiply_vector(w, alpha))
        if two_norm_squared(next_r) < epsilon or iterations > 1000:
            break
        beta_i = two_norm_squared(next_r) / two_norm_squared(current_r)
        next_p = add_vectors(next_r, scalar_multiply_vector(current_p, beta_i))
        current_r = next_r
        current_p = next_p
        current_x = next_x

    return current_x, iterations


def jacobi_preconditioner_cg(ca, ci, cj, b, x_0, epsilon):
    diag_a = []
    diag_a_indx = []
    for indx in range(len(ci)):
        if cj[indx] == ci[indx]:
            diag_a.append(ca[indx])
            diag_a_indx.append(cj[indx])
    inv_diag_a_elements = [1 / a for a in diag_a]

    solution, iterations = preconditioned_conjugate_gradient(
        ca, ci, cj, b, x_0, epsilon, inv_diag_a_elements, diag_a_indx, diag_a_indx
    )

    return solution, iterations


def tridiagonal(ta, ti, tj):
    "Returns diagonal, lower, and upper diagonals for tridiagonal of input matrix."
    diagonal, lower, upper = [0] * max(ti), [0] * (max(ti) - 1), [0] * (max(ti) - 1)

    for i in range(len(ta)):
        row = ti[i] - 1
        column = tj[i] - 1
        element = ta[i]

        if row == column:
            diagonal[row] = element
        elif row == column + 1:
            lower[column] = element
        elif column == row + 1:
            upper[row] = element

    return diagonal, upper, lower


def tridiagonal_lu(ta, ti, tj):
    """Return diag, lower, upper for L, U decomp of matrix"""
    diag, upper, lower = tridiagonal(ta, ti, tj)
    n = len(diag)

    u_upper = [0] * (n - 1)
    u_diag = [0] * n
    l_lower = [0] * (n - 1)

    u_diag[0] = diag[0]
    u_upper[0] = upper[0]

    for i in range(1, n):
        l_lower[i - 1] = lower[i - 1] / u_diag[i - 1]
        u_diag[i] = diag[i] - l_lower[i - 1] * upper[i - 1]
        if i < n - 1:
            u_upper[i] = upper[i]

    return u_diag, l_lower, u_upper


def tridiagonal_lu_coordinate_list(ta, ti, tj):
    """Finds tridiag LU of a, then converts the result into coordinate list."""
    u_diag, l_lower, u_upper = tridiagonal_lu(ta, ti, tj)
    la, li, lj = [], [], []
    ua, ui, uj = [], [], []

    diag_length = len(u_diag)
    # L coord list
    for i in range(diag_length):
        la.append(1)
        li.append(i + 1)
        lj.append(i + 1)
        if i > 0:
            la.append(l_lower[i - 1])
            li.append(i + 1)
            lj.append(i)

    # U coord list
    for i in range(diag_length):
        ua.append(u_diag[i])
        ui.append(i + 1)
        uj.append(i + 1)
        if i < diag_length - 1:
            ua.append(u_upper[i])
            ui.append(i + 1)
            uj.append(i + 2)

    return (la, li, lj), (ua, ui, uj)


def tridiag_preconditioner_cg(ca, ci, cj, b, x_0, epsilon, max_iterations=1000):
    (tla, tli, tlj), (tua, tui, tuj) = tridiagonal_lu_coordinate_list(ca, ci, cj)

    current_r = subtract_vectors(b, sparse_matrix_vector_multiply(ca, ci, cj, x_0))
    current_y = forward_substitution(tla, tli, tlj, current_r)
    current_z = backward_substitution(tua, tui, tuj, current_y)
    current_p = current_z
    current_x = x_0
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        omega = sparse_matrix_vector_multiply(ca, ci, cj, current_p)
        alpha = dot_product(current_r, current_z) / dot_product(current_p, omega)

        next_x = add_vectors(current_x, scalar_multiply_vector(current_p, alpha))
        next_r = subtract_vectors(current_r, scalar_multiply_vector(omega, alpha))

        current_x = next_x

        if two_norm_squared(next_r) < epsilon:
            break

        next_y = forward_substitution(tla, tli, tlj, next_r)
        next_z = backward_substitution(tua, tui, tuj, next_y)

        beta = dot_product(next_r, next_z) / dot_product(current_r, current_z)
        next_p = add_vectors(next_z, scalar_multiply_vector(current_p, beta))

        current_y = next_y
        current_p = next_p
        current_z = next_z
        current_r = next_r

    return current_x, iterations


def incomplete_cholesky_factorization(ca, ci, cj):
    """Find the incomplete cholesky factorization given coordinate list form matrix."""
    la = list(ca)
    li = list(ci)
    lj = list(cj)

    for column in range(1, max(li) + 1):
        diag_index = None
        for i in range(len(la)):
            if li[i] == column and lj[i] == column:
                diag_index = i
                break

        sum_squared_diag = 0.0
        for i in range(len(la)):
            if li[i] == column and lj[i] < column:
                for j in range(len(la)):
                    if li[j] == column and lj[j] == lj[i]:
                        sum_squared_diag += la[j] ** 2
                        break
        la[diag_index] = sqrt(la[diag_index] - sum_squared_diag)

        for i in range(len(la)):
            if li[i] > column and lj[i] == column:
                sum_ik_jk = 0.0
                for k in range(1, column):
                    l_ik = 0.0
                    l_jk = 0.0
                    for j in range(len(la)):
                        if li[j] == li[i] and lj[j] == k:
                            l_ik = la[j]
                        if li[j] == column and lj[j] == k:
                            l_jk = la[j]
                    sum_ik_jk += l_ik * l_jk
                la[i] = (la[i] - sum_ik_jk) / la[diag_index]
    return la, li, lj


def forward_substitution(la, li, lj, r):
    # solve the system Ly=r for a lower triangular matrix
    diag = [0] * len(r)
    y = list(r)

    for one_indx in range(len(li)):
        i = li[one_indx] - 1
        j = lj[one_indx] - 1
        a_ij = la[one_indx]
        if i == j:
            diag[i] = a_ij

    for one_indx in range(len(li)):
        i = li[one_indx] - 1
        j = lj[one_indx] - 1
        a_ij = la[one_indx]
        if j < i:
            y[i] -= a_ij * (y[j] / diag[j])

    for i in range(len(y)):
        y[i] /= diag[i]
    return y


def backward_substitution(lta, lti, ltj, y):
    # solve the system Ltz=y for an upper triangular matrix
    diag = [0] * len(y)
    z = list(y)

    for one_indx in range(len(lti)):
        i = lti[one_indx] - 1
        j = ltj[one_indx] - 1
        a_ij = lta[one_indx]
        if i == j:
            diag[i] = a_ij

    # we need to loop over these elements in reverse, since the matrix is UPPER triangular
    for one_indx in range(len(lti) - 1, 0, -1):
        i = lti[one_indx] - 1
        j = ltj[one_indx] - 1
        a_ij = lta[one_indx]
        if j > i:
            z[i] -= a_ij * (z[j] / diag[j])

    for i in range(len(z)):
        z[i] /= diag[i]
    return z


def incomplete_cholesky_preconditioned_cg(
    ca, ci, cj, b, x_0, epsilon, max_iterations=1000
):
    cholesky, cholesky_rows, cholesky_columns = incomplete_cholesky_factorization(
        ca, ci, cj
    )

    current_r = subtract_vectors(b, sparse_matrix_vector_multiply(ca, ci, cj, x_0))
    # solve: Ly_0=r_0
    current_y = forward_substitution(
        cholesky, cholesky_rows, cholesky_columns, current_r
    )
    # solve: L^{T}z_0=y_0
    current_z = backward_substitution(
        cholesky, cholesky_columns, cholesky_rows, current_y
    )

    current_p = current_z
    current_x = x_0
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        omega = sparse_matrix_vector_multiply(ca, ci, cj, current_p)
        alpha = dot_product(current_r, current_z) / dot_product(current_p, omega)

        next_x = add_vectors(current_x, scalar_multiply_vector(current_p, alpha))
        next_r = subtract_vectors(current_r, scalar_multiply_vector(omega, alpha))

        current_x = next_x

        if two_norm_squared(next_r) < epsilon:
            break

        # solve: Ly_0=r_0
        next_y = forward_substitution(cholesky, cholesky_rows, cholesky_columns, next_r)
        # solve: L^{T}z_0=y_0
        next_z = backward_substitution(
            cholesky, cholesky_columns, cholesky_rows, next_y
        )

        beta = dot_product(next_r, next_z) / dot_product(current_r, current_z)
        next_p = add_vectors(next_z, scalar_multiply_vector(current_p, beta))

        current_y = next_y
        current_p = next_p
        current_z = next_z
        current_r = next_r

    return current_x, iterations


def preconditioned_conjugate_gradient(
    ca, ci, cj, b, x_0, epsilon, inv_ma, inv_mi, inv_mj, max_iterations=1000
):
    current_r = subtract_vectors(b, sparse_matrix_vector_multiply(ca, ci, cj, x_0))
    current_p = sparse_matrix_vector_multiply(inv_ma, inv_mi, inv_mj, current_r)
    current_y = current_p
    current_x = x_0
    iterations = 0

    while iterations <= max_iterations:
        iterations += 1
        omega = sparse_matrix_vector_multiply(ca, ci, cj, current_p)
        alpha = dot_product(current_y, current_r) / dot_product(current_p, omega)
        next_x = add_vectors(current_x, scalar_multiply_vector(current_p, alpha))
        next_r = subtract_vectors(current_r, scalar_multiply_vector(omega, alpha))
        if two_norm_squared(next_r) < epsilon:
            break
        next_y = sparse_matrix_vector_multiply(inv_ma, inv_mi, inv_mj, next_r)
        beta = dot_product(next_y, next_r) / dot_product(current_y, current_r)
        next_p = add_vectors(next_y, scalar_multiply_vector(current_p, beta))

        current_x = next_x
        current_y = next_y
        current_p = next_p
        current_r = next_r

    return current_x, iterations


# CONSTANTS
ca = [4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # fmt: skip
ci = [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8] # fmt: skip
cj = [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 1, 3, 6, 2, 4, 7, 3, 5, 8, 1, 4, 6, 2, 5, 7, 3, 6, 8, 4, 7] # fmt: skip
b = [6, 7, 7, 7, 7, 7, 7, 6]
initial_guess = [0, 0, 0, 0, 0, 0, 0, 0]
epsilon_4 = 10**-4
epsilon_8 = 10**-8

# call the functions
classic_4, classic_4_iterations = conjugate_gradient(
    ca, ci, cj, b, initial_guess, epsilon_4
)
classic_8, classic_8_iterations = conjugate_gradient(
    ca, ci, cj, b, initial_guess, epsilon_8
)

jacobi_4, jacobi_4_iterations = jacobi_preconditioner_cg(
    ca, ci, cj, b, initial_guess, epsilon_4
)
jacobi_8, jacobi_8_iterations = jacobi_preconditioner_cg(
    ca, ci, cj, b, initial_guess, epsilon_8
)

cholesky_4, cholesky_4_iterations = incomplete_cholesky_preconditioned_cg(
    ca, ci, cj, b, initial_guess, epsilon_4
)
cholesky_8, cholesky_8_iterations = incomplete_cholesky_preconditioned_cg(
    ca, ci, cj, b, initial_guess, epsilon_8
)

tridiag_4, tridiag_4_iterations = tridiag_preconditioner_cg(
    ca, ci, cj, b, initial_guess, epsilon_4
)
tridiag_8, tridiag_8_iterations = tridiag_preconditioner_cg(
    ca, ci, cj, b, initial_guess, epsilon_8
)

print("Results")
print("Classic cg")
print(
    f"Classic conjugate gradient converges in {classic_4_iterations} iterations with error tolerance {epsilon_4}: {classic_4}"
)
print(
    f"Classic conjugate gradient converges in {classic_8_iterations} iterations with error tolerance {epsilon_8}: {classic_8}"
)
print("Jacobi")
print(
    f"Jacobi converges in {jacobi_4_iterations} iterations with error tolerance {epsilon_4}: {jacobi_4}"
)
print(
    f"Jacobi converges in {jacobi_8_iterations} iterations with error tolerance {epsilon_8}: {jacobi_8}"
)
print("Cholesky")
print(
    f"Cholesky converges in {cholesky_4_iterations} iterations with error tolerance {epsilon_4}: {cholesky_4}"
)
print(
    f"Cholesky converges in {cholesky_8_iterations} iterations with error tolerance {epsilon_8}: {cholesky_8}"
)
print("Tridiagonal")
print(
    f"Tridiagonal converges in {tridiag_4_iterations} iterations with error tolerance {epsilon_4}: {tridiag_4}"
)
print(
    f"Tridiagonal converges in {tridiag_8_iterations} iterations with error tolerance {epsilon_8}: {tridiag_8}"
)
