import torch
from module_4 import data_setup, engine, model_builder, eval_model, plot_data

BATCH_SIZE = 128
NUM_EPOCHS = 10
SEED_NUM = 42
OUT_FEAT = 4 # number of classes for last layer
DROP_OUT = 0.2 # dopout for pretrined models
LEARNING_RATE = 0.001

classes = ['Plochá', 'Valbová', 'Sedlová', 'Komplexní']

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
    
    print(f"[INFO] Evauluating {model.name} model with test data.")

    cm, cm_np, all_acc, class_acc, precision, recall, f1_score = eval_model.matrix_and_accuracy(
        model, test_loader, device, OUT_FEAT)

    print(f"[RESULT] Celková přesnost: {all_acc}.")

    plot_data.plot_matrix(classes, cm_np, model.name, all_acc, class_acc, NUM_EPOCHS)


def train_effnetb1():
    model = model_builder.create_effnetb1(device, SEED_NUM, DROP_OUT, OUT_FEAT)
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

    print(f"[INFO] Evauluating {model.name} model with test data.")

    cm, cm_np, all_acc, class_acc, precision, recall, f1_score = eval_model.matrix_and_accuracy(
        model, test_loader, device, OUT_FEAT)

    print(f"[RESULT] Celková přesnost: {all_acc}.")

    plot_data.plot_matrix(classes, cm_np, model.name, all_acc, class_acc, NUM_EPOCHS)


def train_buyu():
    model = model_builder.ReplikaBUYU(OUT_FEAT)
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
    train_effnetb1()
    #train_buyu()