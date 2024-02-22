import os
import torch
from torch import nn
from bp_module import data_setup, engine, model_builder, utils
import pandas as pd

from torchvision import transforms
import torchvision.models as models
#from google.colab import drive

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup hyperparameters
NUM_EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.0004

# Setup directories
#drive.mount('/content/gdrive')
#imgs_path = '/content/gdrive/MyDrive/Colab Notebooks/bp/clip'
#label_csv = '/content/gdrive/MyDrive/Colab Notebooks/bp/labels_2ctg.csv'
imgs_path = './clip'
label_csv = './bp_module/labels_2ctg.csv'

# Create pd.dataframe IMG_PATH / LABEL
label_df = pd.read_csv(label_csv, index_col=None)

# Calculate the sizes for train, validation, and test sets
total_size = len(label_df)
train_size = int(0.6 * total_size)
valid_size = int(0.30 * total_size)
test_size = total_size - train_size - valid_size

# Split the dataset
train_data = label_df.iloc[:train_size]
valid_data = label_df.iloc[train_size:train_size + valid_size]
test_data = label_df.iloc[train_size + valid_size:]

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

# Create transforms
data_transform =  transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# Create DataLoaders with help from data_setup.py
train_dataloader, valid_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_data,
    valid_data,
    test_data,
    imgs_path,
    data_transform,
    BATCH_SIZE
 )

# -- model set-up BEGIN

# Create model with help from model_builder.py
#model = model_builder.ReplikaBUYU().to(device)

model = models.vgg16(weights='DEFAULT').to(device)
# Freeze all feture extr. layers
for param in model.features.parameters():
    #print(param)
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=25088,
              out_features=2)).to(device)

# -- model set-up END

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_loader=train_dataloader,
             valid_loader=valid_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name=f"{model.__str__}_b{BATCH_SIZE}_lr{str(LEARNING_RATE).split('.')[-1]}.pth")