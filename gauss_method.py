
# Gauss method without main element

def gauss_method(a, b):
    """Function return solution(x) of a system of linear algebraic equations ax = b"""

    n = len(a)

    for k in range(n):
        for i in range(k + 1, n):
            factor = a[i][k] / a[k][k]
            for j in range(k, n):
                a[i][j] -= a[k][j] * factor
            b[i] -= b[k] * factor

    x = [0] * n
    for k in range(n - 1, -1, -1):
        sum_of_previous_x = 0
        for i in range(k + 1, n):
            sum_of_previous_x += a[k][i] * x[i]
        x[k] = (1 / a[k][k]) * (b[k] - sum_of_previous_x)

    return x


def main():
    """
    a = [[1, -2, 3, -1],
         [2, 3, -4, 4],
         [3, 1, -2, -2],
         [1, -3, 7, 6]]

    b = [6, -7, 9, -7]
    """

    """
    a = [[1, 2],
         [2, 1]]
    b = [-1, 1]
    """

    """
    a = [[3, 4, 2],
         [4, -3, 4],
         [3, -4, 1]]
    b = [9, 5, 0]
    """

    a = [[4, 5, 1, 2, 3],
         [7, 6, 5, 4, 2],
         [2, 8, 2, 3, 9],
         [9, 1, 3, 5, 1],
         [5, 3, 4, 1, 8]]
    b = [9, 34, 30, 8, 15]

    solution = gauss_method(a, b)
    print(solution)


if __name__ == "__main__":
    main()
