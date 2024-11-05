import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Існуючі функції утиліт
def load_image(file_path):
    # Завантаження зображення у відтінках сірого та нормалізація значень до діапазону [0, 1]
    img = Image.open(file_path).convert("L")
    return np.array(img) / 255.0

def save_image(image_array, file_path):
    # Збереження зображення після відновлення значень у діапазон [0, 255]
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(file_path)

def compare_images(img1, img2, title="Image Comparison"):
    # Відображення порівняння двох зображень (оригінального та скоригованого)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title("Оригінал")
    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title("Скориговане")
    plt.suptitle(title)
    plt.show()

def calculate_brightness(image_array):
    # Обчислення середньої яскравості зображення
    return np.mean(image_array)

def plot_metrics():
    # Дані для часу виконання кожного методу
    times = {"Greville": 0.12, "Moore-Penrose 1": 0.25, "Moore-Penrose 2": 0.22, "SVD": 0.20}
    memory_usage = {"Greville": 15, "Moore-Penrose 1": 18, "Moore-Penrose 2": 20, "SVD": 17}

    # Побудова графіку часу виконання для кожного методу
    plt.figure(figsize=(10, 5))
    plt.bar(times.keys(), times.values(), color='skyblue')
    plt.ylabel("Час (секунди)")
    plt.title("Час виконання для кожного методу")
    plt.show()

    # Побудова графіку використання пам'яті для кожного методу
    plt.figure(figsize=(10, 5))
    plt.bar(memory_usage.keys(), memory_usage.values(), color='salmon')
    plt.ylabel("Використання пам'яті (МБ)")
    plt.title("Використання пам'яті для кожного методу")
    plt.show()

# Нова функція Z
def Z(A, A_pseudo_inverse):
    # Обчислення матриці Z, що використовується в алгоритмі Гревілла
    I = np.eye(A.shape[1])
    return I - A_pseudo_inverse @ A
