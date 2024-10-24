from PIL import Image
import numpy as np

def load_image(image_path):

   # завантаження зображення та конвертація його у матрицю.
    return np.array(Image.open(image_path).convert('L'))

def save_image(image, filepath):

    # зберігаємо матрицю як зображення.
    img = Image.fromarray(image.astype(np.uint8))
    img.save(filepath)
