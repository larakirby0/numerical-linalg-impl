"""Implementation of Jacobi, Gauss-Siedel, and SOR Iterative methods."""

# CONSTANTS
from constants import initial_guess, ca, ci, cj, b, epsilon, omega


# SHARED FUNCTIONS
def a_at_loc(i, j, elements, row_indices, column_indices):
    """Input i, j and return (a)ij."""
    # add 1 to i and j
    # because ci and cj row indices use 1 as first index
    i = i + 1
    j = j + 1
    for row, col, a in zip(row_indices, column_indices, elements):
        if row == i and col == j:
            return a
    return 0


def vector_norm(v):
    """Compute ||v||_{inf}."""
    return max([abs(e) for e in v])


def multiply_matrix_and_vector(ca, ci, cj, v):
    """Compute Av."""
    multiplied_vector = []
    for i, elem in enumerate(v):
        a_at_j = []
        for j in range(len(v)):
            a_at_j.append(a_at_loc(i, j, ca, ci, cj))
        elements_to_sum = [elem * j for j in a_at_j]
        new_element = sum(elements_to_sum)
        multiplied_vector.append(new_element)
    return multiplied_vector


def subtract_vectors(x, y):
    """Return x-y, component-wise."""
    if len(x) != len(y):
        raise ValueError("Can't add vectors of different lengths")
    return [x_el - y_el for (x_el, y_el) in zip(x, y)]


def is_finished(current_guess, ca, ci, cj, epsilon) -> bool:
    """If ||b-Ax||/||b|| < epsilon, return True."""
    b_norm = vector_norm(b)
    b_minus_Ax = subtract_vectors(
        b, multiply_matrix_and_vector(ca, ci, cj, current_guess)
    )
    b_minus_Ax_norm = vector_norm(b_minus_Ax)
    print("CURRENT ERROR", (b_minus_Ax_norm / b_norm))
    return (b_minus_Ax_norm / b_norm) < epsilon


# JACOBI
def jacobi(initial_guess=initial_guess, ca=ca, ci=ci, cj=cj, b=b, epsilon=epsilon):
    """Iterate predictions for an initial guess using the Jacobi method."""
    current_guess = initial_guess
    variations = 0
    n = max(ci)  # n to use for iterating over j
    while not is_finished(current_guess, ca, ci, cj, epsilon):
        variations += 1
        new_guess = []
        for i in range(len(current_guess)):
            b_i = b[i]
            a_ii = a_at_loc(i, i, ca, ci, cj)
            aij_xj = 0

            for j in range(n):
                if j == i:
                    continue
                aij_xj += a_at_loc(i, j, ca, ci, cj) * current_guess[j]

            new_guess.append((b_i - aij_xj) / a_ii)
        current_guess = new_guess
    print("Jacobi guess\n", current_guess, "achieved in", variations, "steps")
    return current_guess


# GAUSS-SEIDEL
def gauss_seidel(
    initial_guess=initial_guess, ca=ca, ci=ci, cj=cj, b=b, epsilon=epsilon
):
    """Iterate predictions for an initial guess using the Gauss-Seidel method."""
    current_guess = initial_guess
    variations = 0
    n = max(ci)  # n to use for iterating over j
    while not is_finished(current_guess, ca, ci, cj, epsilon):
        variations += 1
        new_guess = []
        for i in range(len(current_guess)):
            b_i = b[i]
            a_ii = a_at_loc(i, i, ca, ci, cj)
            aij_xj_1_until_i = 0
            aij_xj_i_until_n = 0

            for j in range(i):
                if j == i:
                    continue
                aij_xj_1_until_i += a_at_loc(i, j, ca, ci, cj) * new_guess[j]
            for j in range(i + 1, n):
                aij_xj_i_until_n += a_at_loc(i, j, ca, ci, cj) * current_guess[j]

            new_guess.append((b_i - aij_xj_1_until_i - aij_xj_i_until_n) / a_ii)
        current_guess = new_guess
        print("guess", current_guess)
    print("GS guess\n", current_guess, "achieved in", variations, "steps")
    return current_guess


# SOR
def successive_overrelaxation(
    initial_guess=initial_guess, ca=ca, ci=ci, cj=cj, b=b, epsilon=epsilon, omega=omega
):
    """Iterate predictions for an initial guess using the Successive Overrelaxation method."""
    current_guess = initial_guess
    variations = 0
    n = max(ci)  # n to use for iterating over j
    while not is_finished(current_guess, ca, ci, cj, epsilon):
        variations += 1
        new_guess = []
        for i in range(len(current_guess)):
            b_i = b[i]
            a_ii = a_at_loc(i, i, ca, ci, cj)
            aij_xj_1_until_i = 0
            aij_xj_i_until_n = 0

            for j in range(i):
                if j == i:
                    continue
                aij_xj_1_until_i += a_at_loc(i, j, ca, ci, cj) * new_guess[j]
            for j in range(i + 1, n):
                aij_xj_i_until_n += a_at_loc(i, j, ca, ci, cj) * current_guess[j]

            xi_new = (b_i - aij_xj_1_until_i - aij_xj_i_until_n) / a_ii
            xi_old = current_guess[i]
            weighted_guess = omega * xi_new + (1 - omega) * xi_old
            new_guess.append(weighted_guess)
        current_guess = new_guess
    print("SOR guess\n", current_guess, "achieved in", variations, "steps")
    return current_guess


# calling each function on A and an arbitrary x_0 guess and comparing.
# jacobi_guess = jacobi(initial_guess=initial_guess)
gs_guess = gauss_seidel(initial_guess=initial_guess)
# sor_guess = successive_overrelaxation(initial_guess=initial_guess)

# # trying some worse guesses to see if it takes longer...it does!
# jacobi_guess = jacobi(initial_guess=[.5, .5, .5, .5, .5, -.5, .5, 25])
# gs_guess = gauss_seidel(initial_guess=[.5, .5, .5, .5, .5, -.5, .5, 25])
# sor_guess = successive_overrelaxation(initial_guess=[.5, .5, .5, .5, .5, -.5, .5, 25])
