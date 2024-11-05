import numpy as np
from src.utils import Z


def greville_pseudoinversion(A):
    # Перевірка, чи потрібно транспонувати матрицю A
    is_swap = False
    if A.shape[0] > A.shape[1]:
        is_swap = True
        A = A.T

    # Ініціалізація з першим вектором та скалярним множником
    current_vector = A[0, :].reshape(-1, 1)
    vector_scalar = np.dot(current_vector.T, current_vector)

    # Якщо скаляр дорівнює нулю, повертаємо поточний вектор як псевдообернений
    if vector_scalar == 0:
        A_pseudo_inverse = current_vector
    else:
        A_pseudo_inverse = current_vector / vector_scalar

    # Початкове значення для A_i
    A_i = current_vector.T
    for i in range(1, A.shape[0]):
        # Вибір поточного вектора
        current_vector = A[i, :].reshape(-1, 1)

        # Обчислення матриці Z для поточного A
        Z_A = Z(A_i, A_pseudo_inverse)
        A_i = np.vstack([A_i, current_vector.T])

        # Обчислення знаменника для Z
        denom_Z = np.dot(current_vector.T, np.dot(Z_A, current_vector))

        # Умовна перевірка для коректного оновлення псевдооберненої матриці
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

    # Якщо матрицю було транспоновано, повертаємо її назад
    if is_swap:
        A_pseudo_inverse = A_pseudo_inverse.T

    return A_pseudo_inverse


def moore_penrose_pseudoinversion(A, initial_approximation='method1'):
    # Перевірка на потребу транспонування
    is_swap = False
    if A.shape[0] > A.shape[1]:
        is_swap = True
        A = A.T

    # Задання константи для точності та початкового значення для дельти
    CONST_E = 1e-8
    delta = 10.0
    A_pseudo_inverse_current = np.inf * np.ones(A.shape).T
    A_pseudo_inverse_next = -np.inf * np.ones(A.shape).T

    # Початкова апроксимація псевдооберненої матриці
    if initial_approximation == 'method1':
        A_pseudo_inverse_current = np.linalg.pinv(A)
    elif initial_approximation == 'method2':
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_inv = np.zeros_like(S)
        for i in range(len(S)):
            if S[i] > 1e-10:
                S_inv[i] = 1 / S[i]
        A_pseudo_inverse_current = Vt.T @ np.diag(S_inv) @ U.T

    # Ітераційний процес для обчислення псевдооберненої матриці
    while np.max(np.square(A_pseudo_inverse_current - A_pseudo_inverse_next)) > CONST_E:
        A_pseudo_inverse_current = A_pseudo_inverse_next
        A_pseudo_inverse_next = np.dot(A.T, np.linalg.inv(np.dot(A, A.T) + delta * np.eye(A.shape[0])))
        delta /= 2.0

    # Якщо матриця була транспонована, повертаємо її у відповідний формат
    if is_swap:
        A_pseudo_inverse_next = A_pseudo_inverse_next.T

    return A_pseudo_inverse_next


def svd_pseudoinversion(A):
    # Використання SVD для обчислення псевдооберненої матриці
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.zeros_like(S)
    for i in range(len(S)):
        if S[i] > 1e-10:
            S_inv[i] = 1 / S[i]
    A_pseudo_inverse = Vt.T @ np.diag(S_inv) @ U.T
    return A_pseudo_inverse
