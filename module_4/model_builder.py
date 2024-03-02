import torch
from torch import nn
import torchvision.models as models


def create_vgg16(device, SEED_NUM, DROP_OUT, OUT_FEAT, freez):
    model = models.vgg16(weights='DEFAULT').to(device)

    # freez all feture extr. layers
    if freez:
        print(f"[INFO] FREEZING model grad layers.")
        for param in model.features.parameters():
            #print(param)
            param.requires_grad = False
    else:
        print(f"[INFO] Model is in UNFREEZE mode.")

    torch.manual_seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)

    model.classifier = nn.Sequential(
        nn.Dropout(p=DROP_OUT, inplace=True),
        nn.Linear(in_features=25088,
                out_features=OUT_FEAT)).to(device)
    
    model.name = "VGG_16"
    print(f"[INFO] Created new {model.name} model.")
    
    return model


def create_effnetb1(device, SEED_NUM, DROP_OUT, OUT_FEAT, freez=True):
    weights = models.EfficientNet_B1_Weights.DEFAULT
    model = models.efficientnet_b1(weights=weights).to(device)
    # freez all feture extr. layers
    if freez:
        print(f"[INFO] FREEZING model grad layers.")
        for param in model.features.parameters():
            #print(param)
            param.requires_grad = False
    else:
        print(f"[INFO] Model is in UNFREEZE mode.")

    torch.manual_seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)

    model.classifier = nn.Sequential(
        nn.Dropout(p=DROP_OUT, inplace=True),
        nn.Linear(in_features=1280,
                out_features=OUT_FEAT)).to(device)

    model.name = "EfficientNet_b1"
    print(f"[INFO] Created new {model.name} model.")

    return model


def create_resnet101(device, SEED_NUM, DROP_OUT, OUT_FEAT, freez=True):
    weights = models.ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights).to(device)

    # freez all feture extr. layers
    if freez:
        print(f"[INFO] FREEZING model grad layers.")
        for param in model.parameters():
            #print(param)
            param.requires_grad = False
    else:
        print(f"[INFO] Model is in UNFREEZE mode.")

    torch.manual_seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)

    model.fc = nn.Sequential(
        nn.Dropout(p=DROP_OUT, inplace=True),
        nn.Linear(in_features=2048,
                out_features=OUT_FEAT)).to(device)

    model.name = "ResNet_101"
    print(f"[INFO] Created new {model.name} model.")

    return model