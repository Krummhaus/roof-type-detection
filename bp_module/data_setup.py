import os
import torch
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from typing import Tuple

# NUM_WORKERS = os.cpu_count() # causing errors on local-linux-machine
NUM_WORKERS = 0

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe: pd.DataFrame, root_dir: str, transform=None) -> None:
        self.dataframe = dataframe.copy()
        self.root_dir = root_dir
        self.transform = transform

        # Create class attributes
        # Get all image paths
        self.paths = [os.path.join(self.root_dir, f'{image}.png') for image in self.dataframe['RUIAN_ID']]
        # Create classes and class_to_idx attributes
        self.classes = self.dataframe['LABEL'].unique().tolist()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        img_rgba =  Image.open(image_path)
        img_rgb = img_rgba.convert('RGB')
        return img_rgb

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        label = self.class_to_idx[self.dataframe.iloc[index]['LABEL']]

        if self.transform:
            return self.transform(img), label
        else:
            return img, label

def create_dataloaders(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    imgs_path: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
        train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    train_dataset = CustomDataset(train_data, imgs_path, transform=transform)
    valid_dataset = CustomDataset(valid_data, imgs_path, transform=transform)
    test_dataset = CustomDataset(test_data, imgs_path, transform=transform)

    # Get class names
    class_names = train_dataset.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False, # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader, test_dataloader, class_names