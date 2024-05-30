import random
import numpy as np
import torch
import cv2

def convert_gray_to_rgb(wafer_map):
    output = (wafer_map / np.max(wafer_map) * 255).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True