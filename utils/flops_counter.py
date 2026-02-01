import sys
import os

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import torch
from ptflops import get_model_complexity_info
from models.simple_cnn import SimpleCNN


def count_flops():
    model = SimpleCNN(num_classes=10)

    flops, params = get_model_complexity_info(
        model,
        (3, 32, 32),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    print("Model FLOPs:", flops)
    print("Model Parameters:", params)

    return flops, params


if __name__ == "__main__":
    count_flops()
