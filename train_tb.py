# Train function with Tensor board implemented
import os
import torch
from torch import nn
from bp_module_tb import data_setup, engine, model_builder, utils
import pandas as pd

from torchvision import transforms
import torchvision.models as models


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

# Define seed function
def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_buyu():
    model = model_builder.ReplikaBUYU().to(device)
    model.name = "replika_buyu"

    set_seeds()

    print(f"[INFO] Created new {model.name} model.")

    return model


def create_vgg16():
    model = models.vgg16(weights='DEFAULT').to(device)
    # Freeze all feture extr. layers
    for param in model.features.parameters():
        #print(param)
        param.requires_grad = False

    set_seeds()

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=25088,
                out_features=2)).to(device)
    
    model.name = "vgg_16"
    print(f"[INFO] Created new {model.name} model.")
    
    return model


def create_effnetb1():
    weights = models.EfficientNet_B1_Weights.DEFAULT
    model = models.efficientnet_b1(weights=weights).to(device)
    # Freeze all feture extr. layers
    for param in model.features.parameters():
        #print(param)
        param.requires_grad = False

    set_seeds()

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280,
                out_features=2)).to(device)

    model.name = "effnet_b1"
    print(f"[INFO] Created new {model.name} model.")

    return model

# -- Formulate an experiment ------------
BATCH_SIZE = 128
LEARNING_RATE = 0.001

imgs_path = './clip'
label_csv = './bp_module/labels_2ctg.csv'

exp_models = ["vgg_16", "effnet_b1"]
num_epochs = [10]
train_valid_size = [(0.5, 0.45), (0.3, 0.3)]

set_seeds(seed=42)

exp_num = 0

# 1 Loop troucgh each dataloder setup and create dataloader for it
for val in train_valid_size:
    TRAIN_SIZE = val[0]
    VALID_SIZE = val[1]
    # Setup paths

    # Create pd.dataframe IMG_PATH / LABEL
    label_df = pd.read_csv(label_csv, index_col=None)

    # Calculate the sizes for train, validation, and test sets
    total_size = len(label_df)
    train_size = int(TRAIN_SIZE * total_size)
    valid_size = int(VALID_SIZE * total_size)
    test_size = total_size - train_size - valid_size

    # Split the dataset
    train_data = label_df.iloc[:train_size]
    valid_data = label_df.iloc[train_size:train_size + valid_size]
    test_data = label_df.iloc[train_size + valid_size:]

    train_dataloader, valid_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_data,
        valid_data,
        test_data,
        imgs_path,
        data_transform,
        BATCH_SIZE
    )

    # 2 Loop trough num of epochs
    for epoch in num_epochs:

        # 3 Loop trough each model in setup
        for model_name in exp_models:
            exp_num += 1
            print(f"[INFO] Experiment number: {exp_num}")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] DataLoader: {TRAIN_SIZE=} {VALID_SIZE=}")
            print(f"[INFO] Number of epochs: {num_epochs}")  

            if model_name == "vgg_16":
                model = create_vgg16()
            else:
                model = create_effnetb1()

            # Set loss and optimizer
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

            # Start training with help from engine.py
            engine.train(model=model,
                        train_loader=train_dataloader,
                        valid_loader=valid_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=epoch,
                        device=device,
                        writer=engine.create_writer(experiment_name=f"exper_{exp_num}",
                                       model_name=f"{model_name}_b{BATCH_SIZE}_lr{str(LEARNING_RATE).split('.')[-1]}_data{int(TRAIN_SIZE*100)}to{int(VALID_SIZE*100)}",
                                       extra=f"{epoch}_epochs")) 

            # Save the model with help from utils.py
            utils.save_model(model=model,
                            target_dir="models",
                            model_name=f"{model_name}_b{BATCH_SIZE}_lr{str(LEARNING_RATE).split('.')[-1]}_data{int(TRAIN_SIZE*100)}to{int(VALID_SIZE*100)}.pth")