import numpy as np
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

    matrix[0][0] += 10 ** (-1 * k)

    return matrix


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

    index_of_max_component = np.abs(V[num_of_iterations+1]).argmax()
    lambda1_1 = V[num_of_iterations + 1][index_of_max_component] * np.sign(U[num_of_iterations][index_of_max_component])
    lambda1_2 = np.matmul(V[num_of_iterations + 1].T, U[num_of_iterations]) / np.matmul(U[num_of_iterations].T, U[num_of_iterations])

    index = np.array([abs(V[m][i] - lambda1_1 * U[m - 1][i]) for i in range(n)]).argmax()
    lambda2_1 = (V[m + 1][index] * np.linalg.norm(V[m], ord=np.inf) - lambda1_1 * V[m][index]) / (V[m][index] - lambda1_1 * U[m - 1][index])
    
    index = np.array([abs(V[m][i] - lambda1_2 * U[m - 1][i]) for i in range(n)]).argmax()
    lambda2_2 = (V[m + 1][index] * np.linalg.norm(V[m], ord=np.inf) - lambda1_2 * V[m][index]) / (V[m][index] - lambda1_2 * U[m - 1][index])

    return lambda1_1, lambda1_2, lambda2_1, lambda2_2, U[num_of_iterations].T, V[num_of_iterations + 1]
    

def main():
    A = np.array(generate_matrix(5, -4, 0, 0))
    print(f"A: {A}\n")

    lambda1_1, lambda1_2, lambda2_1, lambda2_2, U_k, V_k_plus_1 = stepennoy(A, 46, m=30)
    print(f"Num of iterations: 46\nlambda1_1: {lambda1_1}\nlambda1_2: {lambda1_2}\nU_k: {U_k}\n")

    lambda1_1, lambda1_2, lambda2_1, lambda2_2, U_k, V_k_plus_1 = stepennoy(A, 47, m=30)
    print(f"Num of iterations: 47\nlambda1_1: {lambda1_1}\nlambda1_2: {lambda1_2}\nU_k: {U_k}\n")

    lambda1_1, lambda1_2, lambda2_1, lambda2_2, U_k, V_k_plus_1 = stepennoy(A, 48, m=30)
    print(f"Num of iterations: 48\nlambda1_1: {lambda1_1}\nlambda1_2: {lambda1_2}\nU_k: {U_k}\n")

    lambda1_1, lambda1_2, lambda2_1, lambda2_2, U_K, V_k_plus_1 = stepennoy(A, 49, m=30)
    print(f"Num of iterations: 49\nlambda1_1: {lambda1_1}\nlambda1_2: {lambda1_2}\nU_k: {U_k}\n")

    lambda1_1, lambda1_2, lambda2_1, lambda2_2, U_k, V_k_plus_1 = stepennoy(A, 50, m=30)
    print(f"Num of iterations: 50\nlambda1_1: {lambda1_1}\nlambda1_2: {lambda1_2}\nU_k: {U_k}\n")
    v1 = V_k_plus_1 - lambda1_1 * U_k.T
    v2 = V_k_plus_1 - lambda1_2 * U_k.T
    print(f"V_k_plus_1 - lambda1_1 * U_k: {v1.T}\nV_k_plus_1 - lambda1_2 * U_k: {v2.T}\n")
    print(f"Norm_1: {np.linalg.norm(v1, ord=np.inf)}\nNorm_2: {np.linalg.norm(v2, ord=np.inf)}\n")
    
    x1 = V_k_plus_1 - lambda1_1 * U_k.T
    x2 = V_k_plus_1 - lambda1_2 * U_k.T
    print(f"x1: {x1.T}\nx2: {x2.T}\n")
    print(f"Num of iterations: 50\nm: 30\nlambda2_1: {lambda2_1}")
    v = np.matmul(A, x1) - lambda2_1 * x1
    print(f"A * x1 - lambda2_1 * x1: {v.T}")
    print(f"Norm: {np.linalg.norm(v, ord=np.inf)}\n")

    lambda1_1, lambda1_2, lambda2_1, lambda2_2, U_k, V_k_plus_1 = stepennoy(A, 50, m=50)
    print(f"Num of iterations: 50\nm: 50\nlambda2_1: {lambda2_1}")
    v = np.matmul(A, x1) - lambda2_1 * x1
    print(f"A * x1 - lambda2_1 * x1: {v.T}")
    print(f"Norm: {np.linalg.norm(v, ord=np.inf)}\n")

    lambda1_1, lambda1_2, lambda2_1, lambda2_2, U_k, V_k_plus_1 = stepennoy(A, 50, m=50)
    print(f"Num of iterations: 50\nm: 50\nlambda2_2: {lambda2_2}")
    v = np.matmul(A, x2) - lambda2_2 * x2
    print(f"A * x2 - lambda2_2 * x2: {v.T}")
    print(f"Norm: {np.linalg.norm(v, ord=np.inf)}\n")


if __name__ == "__main__":
    main()
