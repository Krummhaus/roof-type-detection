import torch
from torch import nn
import torchvision.models as models

class ReplikaBUYU(nn.Module):
    def __init__(self, OUT_FEAT) -> None:
        super().__init__()
        self.OUT_FEAT = OUT_FEAT

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.OUT_FEAT)#,
            #nn.Softmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.global_average_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def create_vgg16(device, SEED_NUM, DROP_OUT, OUT_FEAT):
    model = models.vgg16(weights='DEFAULT').to(device)
    # Freeze all feture extr. layers
    for param in model.features.parameters():
        #print(param)
        param.requires_grad = False

    torch.manual_seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)

    model.classifier = nn.Sequential(
        nn.Dropout(p=DROP_OUT, inplace=True),
        nn.Linear(in_features=25088,
                out_features=OUT_FEAT)).to(device)
    
    model.name = "vgg_16"
    print(f"[INFO] Created new {model.name} model.")
    
    return model


def create_effnetb1(device, SEED_NUM, DROP_OUT, OUT_FEAT):
    weights = models.EfficientNet_B1_Weights.DEFAULT
    model = models.efficientnet_b1(weights=weights).to(device)
    # Freeze all feture extr. layers
    for param in model.features.parameters():
        #print(param)
        param.requires_grad = False

    torch.manual_seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)

    model.classifier = nn.Sequential(
        nn.Dropout(p=DROP_OUT, inplace=True),
        nn.Linear(in_features=1280,
                out_features=OUT_FEAT)).to(device)

    model.name = "effnet_b1"
    print(f"[INFO] Created new {model.name} model.")

    return model