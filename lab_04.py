import math
import numpy as np


def generate_matrix(n, p, q):
    """Generate tridiagonal matrix [n*n]"""

    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = 5 * (i + 1) ** (p / 2)
        matrix[i, 0:i] = 0.01 * np.array([(i + 1) ** (p / 2) + j ** (q / 2) for j in range(0, i)])
        matrix[i, i + 1:n] = 0.01 * np.array([(i + 1) ** (p / 2) + j ** (q / 2) for j in range(i + 1, n)])

    return matrix.round(4)


def get_relative_error(x, x1):
    """Return relative error of x1"""
    return np.linalg.norm(x - x1, ord=np.inf) / np.linalg.norm(x, ord=np.inf)


def get_num_of_iterations(B, b, epsilon):
    """Return number of iterations for simple iterations method"""
    return math.log((epsilon * (1 - np.linalg.norm(B, ord=np.inf))) / np.linalg.norm(b, ord=np.inf), np.linalg.norm(B, ord=np.inf)) - 1


def sim(A, f):
    """Find approximate solution of system of linear equations using simple iterations method"""

    n = len(f)

    # find `B` and `b` for view `x = Bx + b`
    b = np.array([f[i] / A[i, i] for i in range(n)])
    B = np.copy(A)
    for i in range(n):
        for j in range(n):
            if i != j:
                B[i, j] /= -1 * B[i, i]
        B[i, i] = 0

    # initial approximation
    x = np.copy(f)

    # find number of iterations for simple iterations method
    num_of_iterations = math.ceil(get_num_of_iterations(B, b, epsilon=0.0001))

    # find approximate solution
    for i in range(num_of_iterations):
        x = np.matmul(B, x) + b

    return x


def relaxation_method(A, f, omega, epsilon):
    """Find approximate solution of system of linear equations using relaxation method"""

    # initial approximation
    x = np.copy(f)

    n = len(f)

    k_max = 1000
    for k in range(k_max):
        x_previous = np.copy(x)
        for i in range(n):
            x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (f[i] - sum(A[i, :i] * x[:i]) - sum(A[i, i+1:] * x[i+1:]))

        if np.linalg.norm(x - x_previous, ord=np.inf) < epsilon:
            break

    print(f"\nNum of iterations: {k}")
    return x


def main():
    # generate matrix
    A = generate_matrix(10, 1, 1)
    # generate vector
    x = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    np.set_printoptions(linewidth=np.inf)

    print("Matrix A: ")
    print(A)

    print(f"\nVector x: {x}")

    # find `A * x`
    f = np.matmul(A, x)
    print(f"\nf = A * x = {f}")

    # solve equation using simple iterations method
    print("\nSimple iterations method")
    x1 = sim(A, f)
    print(f"\nx1* = {x1}")

    # solve equation using relaxation method
    print("\nRelaxation method")
    x2 = relaxation_method(A, f, 0.5, 0.0001)
    print(f"\nx2* = {x2}")

    # find relative error
    relative_error1 = get_relative_error(x, x1)
    print(f"\nRelative error for solution by simple iterations method: {relative_error1}")
    relative_error2 = get_relative_error(x, x2)
    print(f"\nRelative error for solution by relaxation method: {relative_error2}")


if __name__ == "__main__":
    main()
