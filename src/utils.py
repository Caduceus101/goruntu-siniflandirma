import os
import matplotlib.pyplot as plt
from src.config import TRAIN_PATH, VAL_PATH


def plot_class_distribution():
    class_name_train = os.listdir(TRAIN_PATH)
    class_name_val = os.listdir(VAL_PATH)

    image_count_train = {i: len(os.listdir(os.path.join(TRAIN_PATH, i))) for i in class_name_train}
    image_count_val = {i: len(os.listdir(os.path.join(VAL_PATH, i))) for i in class_name_val}

    sum_train_images = sum(image_count_train.values())
    sum_val_images = sum(image_count_val.values())

    image_count = {'Egitim': sum_train_images, 'Test': sum_val_images}

    fig1, ax1 = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
    ax1.pie(image_count.values(),
            labels=image_count.keys(),
            shadow=True,
            autopct='%1.1f%%',
            startangle=90)
    plt.title("Eğitim ve Test Veri Seti Dağılımı")
    plt.show()