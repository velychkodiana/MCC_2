import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from src.image_processor import load_image, save_image
from src.pseudoinverse_methods import (
    greville_pseudoinversion,
    moore_penrose_pseudoinversion,
    svd_pseudoinversion
)
from src.utils import Z, compare_images, calculate_brightness

# Зберігаємо метрики для побудови графіків
times = {}
memories = {}
brightness_values = {}
pseudoinverse_properties = {}

# Ініціалізація вимірювання продуктивності
def initialize_performance(method_name):
    tracemalloc.start()
    start_time = time.time()
    return start_time, []

# Завершення вимірювання продуктивності
def finalize_performance(method_name, start_time, time_points, mem_points, brightness):
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Зберігаємо результати
    times[method_name] = (start_time, end_time, time_points)
    memories[method_name] = (current / 1024, peak / 1024, mem_points)
    brightness_values[method_name] = brightness

    # Записуємо результати у файл
    with open('/Users/macbookpro/Desktop/МСС/Lab2/output.txt', 'a') as f:
        f.write(f"Метод: {method_name}\n")
        f.write(f"Час виконання: {end_time - start_time} секунд\n")
        f.write(f"Використано пам'яті: {current / 1024} KB (пік: {peak / 1024} KB)\n")
        f.write(f"Яскравість результату: {brightness}\n\n")

# Вимірюємо продуктивність конкретного методу
def measure_performance(method_name, method, X, Y):
    start_time, time_points = initialize_performance(method_name)
    mem_points = []

    # Початкове значення пам'яті та часу
    current, peak = tracemalloc.get_traced_memory()
    mem_points.append(current / 1024)  # Пам'ять на старті
    time_points.append(0)  # Початковий час

    # Виконання методу
    X_pseudo_inverse = method(X)

    # Обчислюємо властивості псевдооберненої матриці
    check_properties(method_name, X, X_pseudo_inverse)

    # Обчислення матриці A на основі псевдоперевернутої матриці X та випадкових шумів
    A = Y @ X_pseudo_inverse + np.random.rand(Y.shape[0], X.shape[0]) @ Z(X_pseudo_inverse, X)
    Y_corrected = A @ X

    # Записуємо час і пам'ять після обчислень
    time_points.append(time.time() - start_time)  # Кінцевий час
    current, peak = tracemalloc.get_traced_memory()
    mem_points.append(current / 1024)  # Пам'ять після виконання

    # Обчислюємо яскравість скоригованого зображення
    brightness = calculate_brightness(Y_corrected)

    finalize_performance(method_name, start_time, time_points, mem_points, brightness)
    return Y_corrected

# Функція для перевірки властивостей псевдооберненої матриці
def check_properties(method_name, A, A_plus):
    prop_1 = np.allclose(A @ A_plus @ A, A)  # AA+A = A
    prop_2 = np.allclose(A_plus @ A @ A_plus, A_plus)  # A+AA+ = A+
    prop_3 = np.allclose(A @ A_plus, (A @ A_plus).T)  # AA+ симетрична
    prop_4 = np.allclose(A_plus @ A, (A_plus @ A).T)  # A+A симетрична
    pseudoinverse_properties[method_name] = (prop_1, prop_2, prop_3, prop_4)

# Функція для побудови графіків
def plot_metrics():
    plt.figure(figsize=(12, 6))

    # Графік часу виконання
    for method_name, (_, _, time_points) in times.items():
        plt.plot(range(len(time_points)), time_points, label=f'{method_name} ', marker='o')

    plt.title('Час виконання методів')
    plt.xlabel('Етап виконання')
    plt.ylabel('Час (с)')
    plt.legend()
    plt.grid()
    plt.show()

    # Графік використання пам'яті
    plt.figure(figsize=(12, 6))
    for method_name, (_, _, mem_points) in memories.items():
        plt.plot(range(len(mem_points)), mem_points, label=f'{method_name} ', marker='o')

    plt.title('Використання пам\'яті методами')
    plt.xlabel('Етап виконання')
    plt.ylabel('Пам\'ять (KB)')
    plt.legend()
    plt.grid()
    plt.show()

    # Графік яскравості результатів
    plt.figure(figsize=(12, 6))
    methods = ["Оригінал", "Greville", "Moore-Penrose 1", "Moore-Penrose 2", "SVD"]
    brightness_values_list = [
        brightness_values["Оригінал"], brightness_values["Greville"],
        brightness_values["Moore-Penrose 1"], brightness_values["Moore-Penrose 2"],
        brightness_values["SVD"]
    ]
    bars = plt.bar(methods, brightness_values_list, color=['gray', 'blue', 'orange', 'green', 'red'])

    # Додання значень на графік
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va='bottom' для розміщення тексту над стовпцем

    plt.title('Яскравість результатів для всіх методів')
    plt.xlabel('Методи')
    plt.ylabel('Яскравість')
    plt.grid(axis='y')
    plt.show()

# Головна функція
def main():
    # Очищення попередніх результатів з файлу
    with open('/Users/macbookpro/Desktop/МСС/Lab2/output.txt', 'w') as f:
        f.write("Результати вимірювань для методів Гревіля, Мура-Пенроуза та SVD:\n\n")

    # Завантаження зображень
    X_path = '/Users/macbookpro/Desktop/МСС/Lab2/x1.bmp'
    Y_path = '/Users/macbookpro/Desktop/МСС/Lab2/y3.bmp'
    result_path_greville = '/Users/macbookpro/Desktop/МСС/Lab2/results/result_greville.bmp'
    result_path_mp = '/Users/macbookpro/Desktop/МСС/Lab2/results/result_mp.bmp'

    X = load_image(X_path)
    X = np.vstack([X, np.ones(X.shape[1])])  # Додаємо рядок одиниць

    Y = load_image(Y_path)

    # Вимірюємо продуктивність методу Гревіля
    Y_greville_corrected = measure_performance("Greville", greville_pseudoinversion, X, Y)
    save_image(Y_greville_corrected, result_path_greville)

    # Вимірюємо продуктивність методу Мура-Пенроуза (два алгоритми)
    Y_mp_corrected_method1 = measure_performance("Moore-Penrose 1",
                                                 lambda X: moore_penrose_pseudoinversion(X, 'method1'), X, Y)
    save_image(Y_mp_corrected_method1, result_path_mp.replace('.bmp', '_method1.bmp'))

    Y_mp_corrected_method2 = measure_performance("Moore-Penrose 2",
                                                 lambda X: moore_penrose_pseudoinversion(X, 'method2'), X, Y)
    save_image(Y_mp_corrected_method2, result_path_mp.replace('.bmp', '_method2.bmp'))

    # Вимірюємо продуктивність методу SVD
    Y_svd_corrected = measure_performance("SVD", svd_pseudoinversion, X, Y)
    save_image(Y_svd_corrected, result_path_mp.replace('.bmp', '_svd.bmp'))

    # Обчислюємо яскравість для оригінального зображення
    original_brightness = calculate_brightness(Y)
    brightness_values["Оригінал"] = original_brightness

    # Порівняння зображень та їх відображення
    compare_images(Y, Y_greville_corrected, title='Результат методу Гревіля')
    plt.show()

    compare_images(Y, Y_mp_corrected_method1, title='Результат методу Мура-Пенроуза 1')
    plt.show()

    compare_images(Y, Y_mp_corrected_method2, title='Результат методу Мура-Пенроуза 2')
    plt.show()

    compare_images(Y, Y_svd_corrected, title='Результат методу SVD')
    plt.show()

    # Побудова графіків
    plot_metrics()

    # Функція для виводу властивостей псевдооберненої матриці
    def display_pseudoinverse_properties():
        # Заголовок
        print("Властивості псевдооберненої матриці для всіх методів:")

        # Властивості
        properties_headers = ["AA+ = A", "A+AA = A+", "AA+ симетрична", "A+A симетрична"]

        # Виводимо таблицю з перевіркою властивостей
        print(f"{'Метод':<20} {' | '.join(properties_headers)}")

        print("-" * (20 + len(' | '.join(properties_headers)) + 3))

        # Виводимо властивості для кожного методу
        for method, properties in pseudoinverse_properties.items():
            formatted_properties = ["True" if prop else "False" for prop in properties]
            print(f"{method:<20} {' | '.join(formatted_properties)}")

    # Приклад використання функції
    display_pseudoinverse_properties()

if __name__ == "__main__":
    main()
