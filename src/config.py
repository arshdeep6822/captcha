DATA_DIR = "/Users/arsh/Desktop/captcha/input/jpg_images"
BATCH_SIZE = 8
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 0
EPOCHS = 10
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")