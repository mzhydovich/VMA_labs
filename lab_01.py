from random import randint
from gauss_method import gauss_method
from gauss_method import gauss_method_with_main_element


def multiply(a, b):
    """Multiply matrix `a` by vector `b`"""
    n = len(a)
    m = len(b)

    x = [0] * n
    for i in range(n):
        xi = 0
        for q in range(m):
            xi += a[i][q] * b[q]
        x[i] = xi

    return x


def print_matrix(a):
    """Print matrix"""
    n = len(a)
    m = len(a[0])
    for i in range(n):
        for q in range(m):
            print(f"{a[i][q]:>10}", end=" ")
        print()


def generate_matrix(n, a, b):
    """Generate matrix [n*n] of random numbers from `a` to `b`"""
    matrix = [0] * n
    for i in range(n):
        row = [0] * n
        for q in range(n):
            row[q] = randint(a, b)
        matrix[i] = row

    return matrix


def generate_vector(n, a, b):
    """Generate vector [n] of random numbers from `a` to `b`"""
    vector = [0] * n
    for i in range(n):
        vector[i] = randint(a, b)

    return vector


def get_norm_of_vector(x):
    """Find the cubic norm of vector"""
    return max([abs(el) for el in x])


def subtract(a, b):
    """Get subtraction b from a"""
    return [(a[i] - b[i]) for i in range(len(a))]


def main():

    # generate matrix
    A = generate_matrix(10, -1000, 1000)
    # generate vector
    x = generate_vector(10, -1000, 1000)

    print("Matrix A: ")
    print_matrix(A)
    print(f"\nVector x: {x}")

    # find `A * x`
    f = multiply(A, x)
    print(f"\nf = A * x = {f}")

    # solve equation using gauss method
    x1 = gauss_method(A, f)
    print(f"\nx1* = {x1}")
    # solve equation using gauss method with main element
    x2 = gauss_method_with_main_element(A, f)
    print(f"x2* = {x2}")

    # find relative error
    relative_error1 = get_norm_of_vector(subtract(x1, x)) / get_norm_of_vector(x)
    print(f"\nRelative error for solution1: {relative_error1}")
    relative_error2 = get_norm_of_vector(subtract(x2, x)) / get_norm_of_vector(x)
    print(f"Relative error for solution2: {relative_error2}")


if __name__ == "__main__":
    main()
