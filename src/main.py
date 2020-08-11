import torch
import argparse
from dynaconf import settings


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

if __name__ == '__main__':
    print(settings.get("data_dir"))
    print(torch.tensor([1, 2, 3], device="cuda"))