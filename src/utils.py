import numpy as np
import matplotlib.pyplot as plt

# функція  для обчислення залишкової різниці Z(A, A_pseudo_inverse)
def Z(A, A_pseudo_inverse):

    return np.eye(A_pseudo_inverse.shape[0]) - np.dot(A_pseudo_inverse, A)

# функція для порівняння зображень
def compare_images(original, transformed, title):

    plt.figure()
    plt.imshow(original.astype(np.uint8), cmap='gray')
    plt.title('Original Image')

    plt.figure()
    plt.imshow(transformed.astype(np.uint8), cmap='gray')
    plt.title(title)
    plt.show()
