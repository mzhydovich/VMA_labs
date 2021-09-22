
# # Gauss method with main element

def gauss_method(a, b):

    n = len(a)
    changed_a = a.copy()

    for k in range(n):
        max_index = changed_a[k].index(max(changed_a[k], key= abs))
        # find max element in row and swap with current
        for c in range(n):
            changed_a[c][max_index], changed_a[c][k] = \
                changed_a[c][k], changed_a[c][max_index]

        for i in range(k + 1, n):
            factor = changed_a[i][k] / changed_a[k][k]
            for j in range(k, n):
                changed_a[i][j] -= changed_a[k][j] * factor
            b[i] -= b[k] * factor

    x = [0] * n
    for k in range(n - 1, -1, -1):
        sum_of_previous_x = 0
        for i in range(k + 1, n):
            sum_of_previous_x += changed_a[k][i] * x[i]
        x[k] = (1 / changed_a[k][k]) * (b[k] - sum_of_previous_x)

    return x


def main():
    """
    a = [[4, 5, 1, 2, 3],
         [7, 6, 5, 4, 2],
         [2, 8, 2, 3, 9],
         [9, 1, 3, 5, 1],
         [5, 3, 4, 1, 8]]
    b = [9, 34, 30, 8, 15]
    """

    """
    a = [[3, 4, 2],
         [4, -3, 4],
         [3, -4, 1]]
    b = [9, 5, 0]
    """

    a = [[1, -2, 3, -1],
         [2, 3, -4, 4],
         [3, 1, -2, -2],
         [1, -3, 7, 6]]

    b = [6, -7, 9, -7]

    solution = gauss_method(a, b)
    print(solution)


if __name__ == "__main__":
    main()
