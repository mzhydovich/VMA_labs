from random import randint


def generate_matrix(n, a, b, k):
    """Generate symmetric matrix [n*n] with elements from `a` to `b`.
    Main diagonal elements are `sum of the other elements in row * -1` and 10^-k added to `a11`"""

    matrix = [[0] * n for i in range(n)]
    for i in range(n):
        for q in range(i):
            matrix[i][q] = matrix[q][i] = randint(a, b)

    for i in range(n):
        matrix[i][i] = -1 * sum(matrix[i])

    matrix1 = [i.copy() for i in matrix]

    matrix[0][0] += 10 ** (-1 * k)
    matrix1[0][0] += 10 ** (-1 * (k + 1))

    return matrix, matrix1


def print_matrix(a):
    """Print matrix"""
    n = len(a)
    m = len(a[0])
    for i in range(n):
        for q in range(m):
            print(f"{a[i][q]:>10}", end=" &")
        print()


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


def LDLt(m):
    """Find LDLt decomposition"""
    a = [i.copy() for i in m]
    n = len(a)
    t = [0] * (n**2)

    # transformate matrix A
    for k in range(n - 1):
        for i in range(k + 1, n):
            t[i] = a[i][k]
            a[i][k] /= a[k][k]
            for j in range(k + 1, i + 1):
                a[i][j] -= a[i][k] * t[j]

    # find matrix L
    l = [[0] * n for i in range(n)]
    for i in range(n):
        l[i][i] = 1
    for i in range(1, n):
        for j in range(i):
            l[i][j] = a[i][j]

    # find matrix D
    d = [[0] * n for i in range(n)]
    for i in range(n):
        d[i][i] = a[i][i]

    return (l, d, transpose(l))


def transpose(m):
    """Transpose matrix"""
    a = [i.copy() for i in m]
    n = len(a)
    for i in range(n):
        for j in range(i):
            a[i][j], a[j][i] = a[j][i], a[i][j]

    return a


def solve(a, b):
    """Solve A*x = b using LDLt decomposition"""
    l, d, l_t = LDLt(a)
    n = len(a)
    x = [0] * n

    # solve L*Y=b
    y = [0] * n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= l[i][j] * y[j]

    # solve D*Lt*x= b
    for i in range(n - 1, -1, -1):
        x[i] = y[i] / d[i][i]
        for j in range(n - 1, i, -1):
            x[i] -= l[j][i] * x[j]

    return x


def get_norm_of_vector(x):
    """Find the cubic norm of vector"""
    return max([abs(el) for el in x])


def get_relative_error(x, x1):
    """Return relative error of x1"""
    subtract = [(x[i] - x1[i]) for i in range(len(x))]
    return get_norm_of_vector(subtract) / get_norm_of_vector(x)


def main():
    n = 5
    m = 4
    k = 0

    # generate matrix
    A1, A2 = generate_matrix(n, -4, 0, k)
    # generate vector
    x = [i for i in range(m, m + n)]  # range(m, m+n-1 + 1)

    print("Matrix A1: ")
    print_matrix(A1)
    print("\nMatrix A2: ")
    print_matrix(A2)
    print(f"\nVector x: {x}")

    # find `A * x`
    b1 = multiply(A1, x)
    b2 = multiply(A2, x)

    print(f"\nb1 = A1 * x = {b1}")
    print(f"\nb2 = A2 * x = {b2}")

    # find LDLt decomposition
    l1, d1, l_t1 = LDLt(A1)
    l2, d2, l_t2 = LDLt(A2)

    print("\nMatrix L1: ")
    print_matrix(l1)
    print("\nMatrix D1: ")
    print_matrix(d1)
    print("\nMatrix Lt1: ")
    print_matrix(l_t1)
    print("\nMatrix L2: ")
    print_matrix(l2)
    print("\nMatrix D2: ")
    print_matrix(d2)
    print("\nMatrix Lt2: ")
    print_matrix(l_t2)

    # find solution
    x1 = solve(A1, b1)
    x2 = solve(A2, b2)
    print(f"\nx1 = {x1}")
    print(f"\nx2 = {x2}")

    # find relative error
    relative_error1 = get_relative_error(x, x1)
    print(f"\nRelative error for solution1: {relative_error1}")
    relative_error2 = get_relative_error(x, x2)
    print(f"Relative error for solution2: {relative_error2}")


if __name__ == "__main__":
    main()
