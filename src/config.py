import os
import torch

DATA_PATH = 'data'
TRAIN_PATH = os.path.join(DATA_PATH, "Egitim")
VAL_PATH = os.path.join(DATA_PATH, "Test")

# Model ve eğitim parametreleri
LENGTH = 1024
BATCH_SIZE = 32

# Eğitim ortamı (Cihaz) kontrolü
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU üzerinde çalışılıyor...")
else:
    device = torch.device("cpu")
    print("CPU üzerinde çalışılıyor...")