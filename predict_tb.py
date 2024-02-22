
# Train function with Tensor board implemented
import os
import torch
from torch import nn
import pandas as pd

from torchvision import transforms
from PIL import Image
from typing import List, Tuple
import torchvision.models as models
import matplotlib
import matplotlib.pyplot as plt


# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

BATCH_SIZE = 128
TRAIN_SIZE = 0.50
VALID_SIZE = 0.45
imgs_path = './clip'
label_csv = './bp_module/labels_2ctg.csv'

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

# Create transforms
data_transform =  transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

print(test_data.head())

# Define seed function
def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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




def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: transforms = None,
                        device: torch.device=device):
    
    
    # 2. Open image
    img_rgba =  Image.open(image_path)
    img = img_rgba.convert('RGB')

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
    plt.savefig(image_path.split('/')[-1])


model = create_vgg16()
model.load_state_dict(torch.load("./models/vgg_16_b128_lr001_data50to45.pth"))


image_list = ['./clip/41623631.png',
    './clip/48037630.png',
    './clip/45592438.png',
    './clip/47071711.png',
    './clip/46086048.png',]

for img in image_list:
    pred_and_plot_image(model=model,
                            image_path=img,
                            class_names=['Flat', 'Other'],
                            image_size=(224, 224))
