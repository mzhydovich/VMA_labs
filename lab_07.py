import numpy as np
from math import sqrt


def f(p, x):
	"""Find p(x)"""

	result = 0
	n = len(p)
	for i in range(n):
		result += p[i] * (x**(n-i-1))

	return result


def f_diff(p, x):
	"""Find p`(x)"""

	p_copy = np.copy(p)
	
	n = len(p_copy)

	for i in range(n-1):
		p_copy[i] *= (n-i-1)

	return f(p_copy[:n-1], x)


def newton(p, start_approximation, epsilon):
	"""Solve equation p(x) = 0 using `start_approximation`"""
	x = start_approximation

	error = 1
	while(abs(error) >= epsilon):
		x_new = x - f(p, x)/f_diff(p, x)
		error = x_new - x
		x = x_new

	return x


def stepennoy(A, num_of_iterations, m):
    """Find the maximum eigenvalue after `num_of_iterations` iterations using Stepennoy method"""

    n = len(A)

    U = []
    V = []

    # initial approximation: U[0] = [1, 0, 0, ..., 0]
    U.append(np.zeros((n, 1), dtype=int))
    U[0][0, 0] = 1

    V.append(np.zeros((n, 1), dtype=int))

    for k in range(1, num_of_iterations + 2):
        V.append(np.matmul(A, U[k - 1]))    
        U.append(np.divide(V[k], np.linalg.norm(V[k], ord=np.inf)))

    j = 0
    ro = sqrt(((V[m][j] * V[m+2][j] * np.linalg.norm(V[m+1], ord=np.inf)) - ((V[m+1][j] ** 2) * np.linalg.norm(V[m], ord=np.inf))) / (U[m-1][j] * V[m+1][j] - U[m][j]**2 * np.linalg.norm(V[m], ord=np.inf)))    
    cosinus = (V[m+2][j] * np.linalg.norm(V[m+1], ord=np.inf) + ro**2 * U[m][j]) / (2 * ro * V[m+1][j]) 
    sinus = sqrt(1 - cosinus**2)

    a = ro * cosinus
    b = ro * sinus

    # find eigenvectors - Re and Im parts
    first_vector_re = V[m + 1]  - a * U[m]
    second_vector_re = V[m + 1]  - a * U[m]
    first_vector_im = V[m + 1]  - ((-1) * b) * U[m]
    second_vector_im = V[m + 1]  - b * U[m]

    return a, b, first_vector_re, first_vector_im, second_vector_re, second_vector_im


def main():

	# create polynom
	A = np.array([1, -60, -944, -34483, 5945475])

	A_matrix = np.array([[60, 944, 34483, -5945475],
						 [1, 0, 0, 0],
						 [0, 1, 0, 0],
						 [0, 0, 1, 0]])

	# create differential
	A_diff = np.array([4, -180, -1888, -34483], dtype=float)

	# solve A`` = 0
	x1 = -4.55335
	x2 = 34.55335
	
	# find solution of A` = 0
	x = newton(A_diff, x1, 0.0001)
	print(f"x of P`() = {x}")

	print(f"P(x) = {f(A, x)}")

	a, b, first_vector_re, first_vector_im, second_vector_re, second_vector_im = stepennoy(A_matrix, 50, 40)
	a = float(a)
	b = float(b)
	print(f"Max eigenvalues of P(x): {a} + i*{b}, {a} - i*{b}")

	print(f"Eigenvector 1: {first_vector_re} + i{first_vector_im}")
	print(f"Eigenvector 2: {second_vector_re} + i{second_vector_im}")
	

if __name__ == "__main__":
	main()
