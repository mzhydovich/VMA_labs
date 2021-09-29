
# Gauss method without main element
def gauss_method(a, b):
    """Function return solution(x) of a system of linear algebraic equations ax = b"""
    n = len(a)   # size of matrix

    # forward stroke
    for k in range(n):
        for i in range(k + 1, n):
            factor = a[i][k] / a[k][k]
            for j in range(k, n):
                a[i][j] -= a[k][j] * factor
            b[i] -= b[k] * factor

    x = [0] * n   # solution

    # reverse stroke
    for k in range(n - 1, -1, -1):
        sum_of_previous_x = 0
        for i in range(k + 1, n):
            sum_of_previous_x += a[k][i] * x[i]
        x[k] = (1 / a[k][k]) * (b[k] - sum_of_previous_x)

    return x


# Gauss method with main element
def gauss_method_with_main_element(a, b):
    """Function return solution(x) of a system of linear algebraic equations ax = b"""
    n = len(a)   # size of matrix

    changed_a = a.copy()
    swaps_x = [i for i in range(n)]   # vector of `x` swaps

    # forward stroke
    for k in range(n):
        # find index of max element
        max_index = changed_a[k].index(max(changed_a[k], key=abs))

        # swap `max element` column with current
        for c in range(n):
            changed_a[c][max_index], changed_a[c][k] = changed_a[c][k], changed_a[c][max_index]
        swaps_x[k], swaps_x[max_index] = swaps_x[max_index], swaps_x[k]

        for i in range(k + 1, n):
            factor = changed_a[i][k] / changed_a[k][k]
            for j in range(k, n):
                changed_a[i][j] -= changed_a[k][j] * factor
            b[i] -= b[k] * factor

    x = [0] * n   # solution

    # reverse stroke
    for k in range(n - 1, -1, -1):
        sum_of_previous_x = 0
        for i in range(k + 1, n):
            sum_of_previous_x += changed_a[k][i] * x[swaps_x[i]]
        x[swaps_x[k]] = (1 / changed_a[k][k]) * (b[k] - sum_of_previous_x)

    return x


def main():
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

    """
    a = [[1, -2, 3, -1],
        [2, 3, -4, 4],
        [3, 1, -2, -2],
        [1, -3, 7, 6]]
    b = [6, -7, 9, -7]
    """

    a = [[4, 5, 1, 2, 3],
         [7, 6, 5, 4, 2],
         [2, 8, 2, 3, 9],
         [9, 1, 3, 5, 1],
         [5, 3, 4, 1, 8]]
    b = [9, 34, 30, 8, 15]

    solution1 = gauss_method(a, b)
    solution2 = gauss_method_with_main_element(a, b)

    print(f"Solution without main element: {solution1}")
    print(f"Solution with main element: {solution2}")


if __name__ == "__main__":
    main()
