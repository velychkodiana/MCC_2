import time
import tracemalloc
from src.image_processor import load_image, save_image
from src.pseudoinverse_methods import greville_pseudoinversion, moore_penrose_pseudoinversion
from src.utils import Z, compare_images
import numpy as np

def measure_performance(method_name, method, X, Y):

    # вимірюємо  час та пам'ять
    tracemalloc.start()
    start_time = time.time()

    X_pseudo_inverse = method(X)
    A = Y @ X_pseudo_inverse + np.random.rand(Y.shape[0], X.shape[0]) @ Z(X_pseudo_inverse, X)
    Y_corrected = A @ X


    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # зберігаємо результати у файл
    with open('/Users/macbookpro/Desktop/МСС/Lab2/output.txt', 'a') as f:
        f.write(f"Метод: {method_name}\n")
        f.write(f"Час виконання: {end_time - start_time} секунд\n")
        f.write(f"Використано пам'яті: {current / 1024} KB (пік: {peak / 1024} KB)\n\n")

    return Y_corrected

def main():
    #додаю зображення
    X_path = '/Users/macbookpro/Desktop/МСС/Lab2/x1.bmp'
    Y_path = '/Users/macbookpro/Desktop/МСС/Lab2/y3.bmp'
    result_path_greville = '/Users/macbookpro/Desktop/МСС/Lab2/results/result_greville.bmp'
    result_path_mp = '/Users/macbookpro/Desktop/МСС/Lab2/results/result_mp.bmp'

    X = load_image(X_path)
    X = np.vstack([X, np.ones(X.shape[1])])  # Додаємо одиничний рядок

    Y = load_image(Y_path)

    # метод Гревіля
    Y_greville_corrected = measure_performance("Greville", greville_pseudoinversion, X, Y)
    save_image(Y_greville_corrected, result_path_greville)

    # метод Мура-Пенроуза
    Y_mp_corrected = measure_performance("Moore-Penrose", moore_penrose_pseudoinversion, X, Y)
    save_image(Y_mp_corrected, result_path_mp)

    # порівнюємо
    compare_images(Y, Y_greville_corrected, title='Greville Method Result')
    compare_images(Y, Y_mp_corrected, title='Moore-Penrose Method Result')

if __name__ == "__main__":
    # видалення даних з файлу перед записом нових результатів
    with open('/Users/macbookpro/Desktop/МСС/Lab2/output.txt', 'w') as f:
        f.write("Результати вимірювань для методів Гревіля та Мура-Пенроуза:\n\n")

    main()
