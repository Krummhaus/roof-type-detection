import torch
from module_4 import data_setup, engine, model_builder

BATCH_SIZE = 128
NUM_EPOCHS = 5
SEED_NUM = 42
OUT_FEAT = 4 # number of classes for last layer
DROP_OUT = 0.2 # dopout for pretrined models
LEARNING_RATE = 0.001

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

# Data preparation
data_dir = './clip4'
train_loader, val_loader, test_loader, class_names = data_setup.create_dataloader(data_dir,BATCH_SIZE, SEED_NUM)


def train_vgg16():
    model = model_builder.create_vgg16(device, SEED_NUM, DROP_OUT, OUT_FEAT)
    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)
    engine.train(model=model,
                train_loader=train_loader,
                valid_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)


if __name__ == '__main__':
    train_vgg16()