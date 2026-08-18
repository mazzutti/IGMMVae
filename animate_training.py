import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from run_pipeline import get_plot_frame, step1_train_best_model

if __name__ == "__main__":
    step1_train_best_model()
