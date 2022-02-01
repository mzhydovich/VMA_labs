import math
from random import random
import numpy as np


def generate_matrix(n, lft, rht):
    """Generate tridiagonal matrix [n*n] with numbers from `lft` to `rht`"""

    return np.random.randint(lft, rht, size=(n, n))


def danilevskiy_method(A, need_matrices=False):
	"""Find canonical form of frobenius using Danilevskiy method"""

	n = len(A)
	
	matrices = [0] * (n-1)

	for k in range(n-2, -1, -1):
		M = np.eye(n, n)
		M[k] = np.ones((1, n))

		# check zero elements
		while A[k+1, k] > -0.00000001 and A[k+1, k] < 0.00000001:
			print("\nmain element = 0\nGenerate new matrix: ")
			A = generate_matrix(n, -50, 50)
			print(A)
			print('\n')

		M[k] /= A[k+1, k]
		M[k] *= ((-1) * A[k+1])
		M[k, k] /= ((-1) * A[k+1, k])

		# save matrices `M_i`
		matrices[k] = M

		# A = M^-1 * A * M
		A = np.matmul(np.matmul(np.linalg.inv(M), A), M)


	if need_matrices:
		return A.round(), matrices
	else:
		return A.round()


def main():
    # generate matrix
    n = 4
    A = generate_matrix(n, -50, 50)
    #A = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])

    np.set_printoptions(linewidth=np.inf)

    # output
    print("Matrix A: ")
    print(A)

    F, matrices = danilevskiy_method(A, need_matrices=True)
    
    print(f"\nСanonical form of frobenius: \n{F}")
    for i in range(n-1):
    	print(f"\nM_{i+1}: \n{matrices[i]}\n")

    print(f"\nP_1 = {F[0, 0]}\n")

    print(f"Sp A = {sum([A[i, i] for i in range(n)])}\n")


if __name__ == "__main__":
    main()
