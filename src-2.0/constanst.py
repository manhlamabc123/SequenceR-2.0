import torch

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1
SAVE_EVERY = 10000
DATA_DIR = "../data/"
MODEL_PATH = "."