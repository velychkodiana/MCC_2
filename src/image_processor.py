import numpy as np
from PIL import Image

# завантаження зображення та конвертація його у матрицю (у формат numpy L)
def load_image(path):
    img = Image.open(path).convert("L")  # Конвертуємо у відтінки сірого
    return np.array(img)

# зберігаємо матрицю як зображення
def save_image(array, path):
    img = Image.fromarray(array.astype(np.uint8))
    img.save(path)
