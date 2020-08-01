from dynaconf import settings
import torch

if __name__ == '__main__':
    print(settings.get("data_dir"))
    print(torch.tensor([1, 2, 3], device="cuda"))