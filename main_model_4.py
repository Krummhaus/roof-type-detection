import torch
from module_4 import data_setup, engine, model_builder, eval_model, plot_data
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

def train_vgg16(NUM_EPOCHS=5,
                OUT_FEAT=4,
                DROP_OUT=0.0,
                LEARNING_RATE=0.001
                ):
    drop, lr = str(DROP_OUT).split('.')[-1], str(LEARNING_RATE).split('.')[-1]
    setup = f"seed{SEED_NUM}_ep{NUM_EPOCHS}_drop{drop}_lr{lr}"
    model = model_builder.create_vgg16(device, SEED_NUM, DROP_OUT, OUT_FEAT)
    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)
    results = engine.train(model=model,
                train_loader=train_loader,
                valid_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device,
                writer=SummaryWriter(log_dir=os.path.join("runs", timestamp, model.name, setup))
                )
    
    print(f"[INFO] Evauluating {model.name} model with test data.")

    cm, cm_np, all_acc, class_acc, precision, recall, f1_score = eval_model.matrix_and_accuracy(
        model,
        test_loader,
        device,
        OUT_FEAT,
        class_names,
        writer=SummaryWriter(log_dir=os.path.join("runs", timestamp, model.name, setup))
        )

    print(f"[RESULT] Celková přesnost: {all_acc}.")

    plot_data.plot_acc_n_loss(results, model.name, setup, timestamp)
    plot_data.plot_matrix(class_names, cm_np, model.name, all_acc,
                          class_acc, setup, timestamp)
    plot_data.plot_textinfo(class_names, all_acc, class_acc,
                            precision, recall, f1_score,
                            model.name, setup, timestamp)


def train_effnetb1(NUM_EPOCHS=5,
                OUT_FEAT=4,
                DROP_OUT=0.0,
                LEARNING_RATE=0.001
                ):
    drop, lr = str(DROP_OUT).split('.')[-1], str(LEARNING_RATE).split('.')[-1]
    setup = f"seed{SEED_NUM}_ep{NUM_EPOCHS}_drop{drop}_lr{lr}"
 
    model = model_builder.create_effnetb1(device, SEED_NUM, DROP_OUT, OUT_FEAT)
    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)
    results = engine.train(model=model,
                train_loader=train_loader,
                valid_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device,
                writer=SummaryWriter(log_dir=os.path.join("runs", timestamp, model.name, setup))
                )

    print(results)
    print(f"[INFO] Evauluating {model.name} model with test data.")

    cm, cm_np, all_acc, class_acc, precision, recall, f1_score = eval_model.matrix_and_accuracy(
        model,
        test_loader,
        device,
        OUT_FEAT,
        class_names,
        writer=SummaryWriter(log_dir=os.path.join("runs", timestamp, model.name, setup))
        )

    print(f"[RESULT] Celková přesnost: {all_acc}.")

    plot_data.plot_acc_n_loss(results, model.name, setup, timestamp)
    plot_data.plot_matrix(class_names, cm_np, model.name, all_acc,
                          class_acc, setup, timestamp)
    plot_data.plot_textinfo(class_names, all_acc, class_acc,
                            precision, recall, f1_score,
                            model.name, setup, timestamp)


def train_resnet101(NUM_EPOCHS=5,
                OUT_FEAT=4,
                DROP_OUT=0.0,
                LEARNING_RATE=0.001
                ):
    drop, lr = str(DROP_OUT).split('.')[-1], str(LEARNING_RATE).split('.')[-1]
    setup = f"seed{SEED_NUM}_ep{NUM_EPOCHS}_drop{drop}_lr{lr}"
 
    model = model_builder.create_resnet101(device, SEED_NUM, DROP_OUT, OUT_FEAT)
    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)
    results = engine.train(model=model,
                train_loader=train_loader,
                valid_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device,
                writer=SummaryWriter(log_dir=os.path.join("runs", timestamp, model.name, setup))
                )

    print(f"[INFO] Evauluating {model.name} model with test data.")

    cm, cm_np, all_acc, class_acc, precision, recall, f1_score = eval_model.matrix_and_accuracy(
        model,
        test_loader,
        device,
        OUT_FEAT,
        class_names,
        writer=SummaryWriter(log_dir=os.path.join("runs", timestamp, model.name, setup))
        )

    print(f"[RESULT] Celková přesnost: {all_acc}.")

    plot_data.plot_acc_n_loss(results, model.name, setup, timestamp)
    plot_data.plot_matrix(class_names, cm_np, model.name, all_acc,
                          class_acc, setup, timestamp)
    plot_data.plot_textinfo(class_names, all_acc, class_acc,
                            precision, recall, f1_score,
                            model.name, setup, timestamp)

def run_experiment_1():
    drop = [0.0, 0.15]
    epch = [7, 13]
    rate = [0.001]
    for dr in drop:
        for ep in epch:
            for lr in rate:
                train_effnetb1(DROP_OUT=dr, NUM_EPOCHS=ep, LEARNING_RATE=lr)
                train_vgg16(DROP_OUT=dr, NUM_EPOCHS=ep, LEARNING_RATE=lr)
                train_resnet101(DROP_OUT=dr, NUM_EPOCHS=ep, LEARNING_RATE=lr)


if __name__ == '__main__':
    BATCH_SIZE = 128
    SEED_NUM = 42

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device=}")

    # Data preparation
    data_dir = './clip7'
    train_loader, val_loader, test_loader, class_names = data_setup.create_dataloader(data_dir,BATCH_SIZE, SEED_NUM)
    # Until i rename /clip4 folder calasses, I must overide
    #class_names = ['Plochá', 'Valbová', 'Sedlová', 'Komplexní']

    #train_effnetb1(NUM_EPOCHS=5)
    train_vgg16(NUM_EPOCHS=7)
    #train_resnet152()
    #run_experiment_1()
