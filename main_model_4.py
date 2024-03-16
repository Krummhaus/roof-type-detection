import torch
from module_4 import data_setup, engine, model_builder, eval_model, plot_data
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from collections import defaultdict

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

def train_vgg16(SEED_NUM, train_loader, val_loader, test_loader, class_names,
                NUM_EPOCHS=5,
                OUT_FEAT=4,
                DROP_OUT=0.0,
                LEARNING_RATE=0.001, freez=True
                ):
    drop, lr = str(DROP_OUT).split('.')[-1], str(LEARNING_RATE).split('.')[-1]
    setup = f"seed{SEED_NUM}_ep{NUM_EPOCHS}_drop{drop}_lr{lr}"

    model = model_builder.create_vgg16(device,
                                       SEED_NUM,
                                       DROP_OUT,
                                       OUT_FEAT,
                                       freez)
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


def train_effnetb1(SEED_NUM, train_loader, val_loader, test_loader, class_names,
                   NUM_EPOCHS=5,
                   OUT_FEAT=4,
                   DROP_OUT=0.0,
                   LEARNING_RATE=0.001, freez=True
                   ):
    drop, lr = str(DROP_OUT).split('.')[-1], str(LEARNING_RATE).split('.')[-1]
    setup = f"seed{SEED_NUM}_ep{NUM_EPOCHS}_drop{drop}_lr{lr}"
 
    model = model_builder.create_effnetb1(device,
                                          SEED_NUM,
                                          DROP_OUT,
                                          OUT_FEAT,
                                          freez)
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


def train_resnet101(SEED_NUM, train_loader, val_loader, test_loader, class_names,
                    NUM_EPOCHS=5,
                    OUT_FEAT=4,
                    DROP_OUT=0.0,
                    LEARNING_RATE=0.001, freez=True
                    ):
    drop, lr = str(DROP_OUT).split('.')[-1], str(LEARNING_RATE).split('.')[-1]
    setup = f"seed{SEED_NUM}_ep{NUM_EPOCHS}_drop{drop}_lr{lr}"
 
    model = model_builder.create_resnet101(device,
                                           SEED_NUM,
                                           DROP_OUT,
                                           OUT_FEAT,
                                           freez)
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

def count_images_per_class(loader):
    class_counts = defaultdict(int)
    for images, labels in loader:
        for label in labels:
            class_counts[label.item()] += 1
    return class_counts


def img_stat_per_loader_class(BATCH_SIZE, SEED_NUM=13):
    data_dir = './clip7'
    train_loader, val_loader, test_loader, class_names = data_setup.create_dataloader(data_dir,BATCH_SIZE, SEED_NUM)
    # Count images per class in train_loader
    train_class_counts = count_images_per_class(train_loader)

    # Count images per class in val_loader
    val_class_counts = count_images_per_class(val_loader)

    # Count images per class in test_loader
    test_class_counts = count_images_per_class(test_loader)

    # Print counts for each class in train_loader
    print("Train Loader:")
    for class_idx, class_name in enumerate(class_names):
        print(f"{class_name}: {train_class_counts[class_idx]} images")

    # Print counts for each class in val_loader
    print("\nValidation Loader:")
    for class_idx, class_name in enumerate(class_names):
        print(f"{class_name}: {val_class_counts[class_idx]} images")

    # Print counts for each class in test_loader
    print("\nTest Loader:")
    for class_idx, class_name in enumerate(class_names):
        print(f"{class_name}: {test_class_counts[class_idx]} images")

def run_experiment_1(SEED_NUM, train_loader, val_loader, test_loader, class_names):
    drop = [0.2]
    epch = [21]
    rate = [0.001]
    for dr in drop:
        for ep in epch:
            for lr in rate:
                #train_effnetb1(SEED_NUM, train_loader, val_loader, test_loader, class_names, DROP_OUT=dr, NUM_EPOCHS=ep, LEARNING_RATE=lr)
                train_vgg16(SEED_NUM, train_loader, val_loader, test_loader, class_names, DROP_OUT=dr, NUM_EPOCHS=ep, LEARNING_RATE=lr)
                train_resnet101(SEED_NUM, train_loader, val_loader, test_loader, class_names, DROP_OUT=dr, NUM_EPOCHS=ep, LEARNING_RATE=lr)

def main_experimet(BATCH_SIZE):
    seeds = [13]
    for seed in seeds:
        SEED_NUM = seed

    # Data preparation
        data_dir = './clip7'
        train_loader, val_loader, test_loader, class_names = data_setup.create_dataloader(data_dir,BATCH_SIZE, SEED_NUM)
        # Until i rename /clip4 folder calasses, I must overide
        #class_names = ['Plochá', 'Valbová', 'Sedlová', 'Komplexní']

        #train_effnetb1(NUM_EPOCHS=5)
        #train_vgg16(NUM_EPOCHS=7, DROP_OUT=0.15, freez=False)
        #train_resnet152()
        run_experiment_1(SEED_NUM, train_loader, val_loader, test_loader, class_names)


if __name__ == '__main__':
    BATCH_SIZE = 128
    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device=}")

    #img_stat_per_loader_class(BATCH_SIZE, SEED_NUM=13)
    main_experimet(BATCH_SIZE)

