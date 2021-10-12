from . import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os

def get_dataloader(rootDir, transforms, batchSize, shuffle=True):
    ds = datasets.ImageFolder(root=rootDir, transform=transforms)
    loader = DataLoader(ds, batch_size=batchSize,
                shuffle=shuffle,
                num_workers=os.cpu_count(),
                pin_memory=True if config.DEVICE == "cuda" else False)
    return (ds, loader)
    