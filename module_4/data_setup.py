import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

class ConvertToRGB(object):
    def __call__(self, img):
        # Convert RGBA image to RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img


data_transform =  transforms.Compose([
    transforms.Resize((224, 224)),
    ConvertToRGB(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def create_dataloader(data_dir,BATCH_SIZE, SEED_NUM):
    my_data = datasets.ImageFolder(root=data_dir, # target folder of images
                                    transform=data_transform, # transforms to perform on data (images)
                                    target_transform=None)

    class_names = my_data.classes

    # Define the ratios for train, validation, and test sets
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    total_data = len(my_data)

    # Calculate sizes for each set
    train_size = int(train_ratio * total_data)
    val_size = int(val_ratio * total_data)
    test_size = total_data - train_size - val_size

    # Set random seed for reproducible split
    generator = torch.Generator().manual_seed(SEED_NUM)

    # Split the data into training, validation, and testing sets
    train_data, val_data, test_data = random_split(my_data, [train_size, val_size, test_size], generator=generator)

    # Create DataLoader for training and testing sets
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, class_names