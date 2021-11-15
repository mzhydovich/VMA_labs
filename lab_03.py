
def generate_matrix(n, p, q):
	"""Generate tridiagonal matrix [n*n]"""

	matrix = [[0] * n for i in range(n)]

	matrix[0][0] = 5
	matrix[0][1] = 2
	matrix[n - 1][n - 2] = (n) ** (p / 2) + (n - 2) ** (q / 2)
	matrix[n - 1][n - 1] = 5 * (n) ** (p / 2)

	for i in range(1, n - 1):
		matrix[i][i] = 5 * (i + 1) ** (p / 2)
		matrix[i][i - 1] = (i + 1) ** (p / 2) + (i - 1) ** (q / 2)
		matrix[i][i + 1] = (i + 1) ** (p / 2) + (i + 1) ** (q / 2)
			
	return matrix


def print_matrix(a):
	"""Print matrix"""
	n = len(a)
	m = len(a[0])
	for i in range(n):
		for q in range(m):
			print(f"{a[i][q]:>20}", end=" ")
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


def get_norm_of_vector(x):
    """Find the cubic norm of vector"""
    return max([abs(el) for el in x])


def get_relative_error(x, x1):
    """Return relative error of x1"""
    subtract = [(x[i] - x1[i]) for i in range(len(x))]
    return get_norm_of_vector(subtract) / get_norm_of_vector(x)


def progonka(A, b, need_matrix=False):
	"""Solve system of linear equations using tridiagonal matrix algorithm"""

	n = len(A)
	y = [0] * n
	ksi = [0] * n
	eta = [0] * n

	# find coefficients
	ksi[n - 1] = -1 * A[n - 1][n - 2] / A[n - 1][n - 1]
	eta[n - 1] = b[n - 1] / A[n - 1][n - 1]

	for i in range(n - 2, 0, -1):
		tmp = A[i][i] + A[i][i + 1] * ksi[i + 1]
		ksi[i] = -1 * A[i][i - 1] / tmp
		eta[i] = (b[i] - A[i][i + 1] * eta[i + 1]) / tmp
	eta[0] = (b[0] - A[0][1] * eta[1]) / (A[0][0] + A[0][1] * ksi[1])

	# find solution
	y[0] = eta[0]
	for i in range(n - 1):
		y[i + 1] = ksi[i + 1] * y[i] + eta[i + 1]


	# get matrix after direct run
	if need_matrix is True:
		matrix = [[0] * n for i in range(n)]
		matrix[0][0] = 1
		for i in range(1, n):
			matrix[i][i - 1] = -1 * ksi[i]
			matrix[i][i] = 1
		v = [eta[i] for i in range(n)]

		return matrix, v, y

	return y


def main():
	# generate matrix
    A = generate_matrix(10, 1, 1)
    # generate vector
    x = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    print("Matrix A: ")
    print_matrix(A)
    print(f"\nVector x: {x}")

    # find `A * x`
    f = multiply(A, x)
    print(f"\nf = A * x = {f}")

    # solve equation using progonka
    A_after, f_after, x1 = progonka(A, f, need_matrix=True)
    print(f"\nx* = {x1}\n")
    print("Matrix A after: ")
    print_matrix(A_after)
    print(f"\nVector f after: {f_after}")
   
    # find relative error
    relative_error = get_relative_error(x, x1)
    print(f"\nRelative error for solution: {relative_error}")
	
	
if __name__ == "__main__":
	main()
