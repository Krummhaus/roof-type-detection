# Import knihoven pro zpracoví dat
import pandas as pd
import os
# from google.colab import drive

# Import PyTorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as F
from PIL import Image
from typing import Tuple, List, Dict, Optional
import random

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Check versions
# Note: your PyTorch version shouldn't be lower than 1.10.0 and torchvision version shouldn't be lower than 0.11
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"

cnn_imgs_path = './clip'
label_csv = './bp_module/labels_2ctg.csv'




import pandas as pd
label_df = pd.read_csv(label_csv, index_col=None)

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
        
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

from tqdm.auto import tqdm

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=valid_dataloader, # Pozor opet validacni dataloader
            loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

# Calculate the sizes for train, validation, and test sets
total_size = len(label_df)
train_size = int(0.7 * total_size)
valid_size = int(0.2 * total_size)
test_size = total_size - train_size - valid_size

# Split the dataset
train_data = label_df.iloc[:train_size]
valid_data = label_df.iloc[train_size:train_size + valid_size]
test_data = label_df.iloc[train_size + valid_size:]

# Display the number of samples in each set
print(f"Train set size: {len(train_data)} samples")
print(train_data.head())
print(f"Validation set size: {len(valid_data)} samples")
print(f"Test set size: {len(test_data)} samples")

# Definice ransformace obrázku na rozměr 224x224px a normalizace
my = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

vgg16 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_data, cnn_imgs_path, transform=vgg16)
valid_dataset = CustomDataset(valid_data, cnn_imgs_path, transform=vgg16)
test_dataset = CustomDataset(test_data, cnn_imgs_path, transform=vgg16)

#dir(DataLoader)
BATCH_SIZE = 32
NUM_WORKERS = 0
train_dataloader = DataLoader(dataset=train_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS,
                         shuffle=True)

valid_dataloader = DataLoader(dataset=train_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS,
                         shuffle=False)

test_dataloader = DataLoader(dataset=train_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS,
                         shuffle=False)

train_dataloader, valid_dataloader, test_dataloader

vgg16 = models.vgg16(weights='DEFAULT').to(device)
# Freeze all feture extr. layers
for param in vgg16.features.parameters():
    #print(param)
    param.requires_grad = False

torch.manual_seed(42)
torch.cuda.manual_seed(42)

vgg16.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=25088,
              out_features=2)).to(device)

NUM_EPOCHS = 30
# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=vgg16.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Train model_0
vgg16_results = train(model=vgg16,
                        train_dataloader=train_dataloader,
                        test_dataloader=valid_dataloader, # Zde pouzivam validacni sadu na testovani !!
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
