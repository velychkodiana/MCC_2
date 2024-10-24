import numpy as np
from src.utils import Z

def greville_pseudoinversion(A):

    # знаходження псевдооберненої матриці методом Гревіля
    is_swap = False
    if A.shape[0] > A.shape[1]:
        is_swap = True
        A = A.T

    current_vector = A[0, :].reshape(-1, 1)
    vector_scalar = np.dot(current_vector.T, current_vector)

    if vector_scalar == 0:
        A_pseudo_inverse = current_vector
    else:
        A_pseudo_inverse = current_vector / vector_scalar

    A_i = current_vector.T
    for i in range(1, A.shape[0]):
        current_vector = A[i, :].reshape(-1, 1)
        Z_A = Z(A_i, A_pseudo_inverse)
        A_i = np.vstack([A_i, current_vector.T])
        denom_Z = np.dot(current_vector.T, np.dot(Z_A, current_vector))

        if denom_Z > 0:
            A_pseudo_inverse = np.hstack([
                A_pseudo_inverse - (np.dot(Z_A, np.dot(current_vector, current_vector.T)) @ A_pseudo_inverse) / denom_Z,
                np.dot(Z_A, current_vector) / denom_Z
            ])
        else:
            R_A = np.dot(A_pseudo_inverse, A_pseudo_inverse.T)
            denom_R = 1 + np.dot(current_vector.T, np.dot(R_A, current_vector))
            A_pseudo_inverse = np.hstack([
                A_pseudo_inverse - (np.dot(R_A, np.dot(current_vector, current_vector.T)) @ A_pseudo_inverse) / denom_R,
                np.dot(R_A, current_vector) / denom_R
            ])

    if is_swap:
        A_pseudo_inverse = A_pseudo_inverse.T

    return A_pseudo_inverse

def moore_penrose_pseudoinversion(A):

    # знаходження псевдооберненої матриці методом Мура-Пенроуза

    is_swap = False
    if A.shape[0] > A.shape[1]:
        is_swap = True
        A = A.T

    CONST_E = 1e-8
    delta = 10.0
    A_pseudo_inverse_current = np.inf * np.ones(A.shape).T
    A_pseudo_inverse_next = -np.inf * np.ones(A.shape).T

    while np.max(np.square(A_pseudo_inverse_current - A_pseudo_inverse_next)) > CONST_E:
        A_pseudo_inverse_current = A_pseudo_inverse_next
        A_pseudo_inverse_next = np.dot(A.T, np.linalg.inv(np.dot(A, A.T) + delta * np.eye(A.shape[0])))
        delta /= 2.0

    if is_swap:
        A_pseudo_inverse_next = A_pseudo_inverse_next.T

    return A_pseudo_inverse_next
