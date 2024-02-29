import numpy as np
import matplotlib.pyplot as plt


def plot_matrix(classes, matrix, model_name, all_acc, class_acc, setup, timestamp):
    # Plot the confusion matrix with annotations
    accuracy_values = class_acc.numpy()
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, [f"{class_name}\n{accuracy*100:.2f} %" for class_name, accuracy in zip(classes, accuracy_values)], rotation=45)
    plt.yticks(tick_marks, classes)

    # Add numbers inside the cells
    #for i in range(matrix.shape[0]):
        #for j in range(matrix.shape[1]):
            #plt.text(j, i, str(matrix[i, j]), horizontalalignment='center', verticalalignment='center')
    
    # Calculate intensity of blue for each cell
    max_val = matrix.max()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cell_val = matrix[i, j]
            color = 'black' if cell_val < max_val / 2 else 'white'  # Choose black or white based on intensity
            plt.text(j, i, str(cell_val), horizontalalignment='center', verticalalignment='center', color=color)

    #plt.title(f'Confusion Matrix')
    #plt.suptitle(f"Vyhodnocení modelu '{model_name}':\nCelková přesnost modelu: {all_acc * 100:.2f} %",
    #             x=0.5, y=0.98, fontsize=11, ha='center', va='top')

    # Adjust the subplot to give some space between the plot and the suptitle
    plt.subplots_adjust(top=0.85)
    #plt.colorbar()
    #plt.text(0, 1, "Additional information goes here", horizontalalignment='left', verticalalignment='center', fontsize=10)
    plt.xlabel('Prediḱovaný typ střechy / Přesnost')
    plt.ylabel('Skutečný typ střechy')
    plt.tight_layout()
    # Show the plot
    #plt.show()
    plt.savefig(f"./plots/plot_{timestamp}_{model_name}_{setup}_cm.png")
    print(f"[INFO] Figurew 'plot_{timestamp}_{model_name}_{setup}_cm.png' saved.")


def plot_acc_n_loss(results, model_name, setup, timestamp):
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(1, len(results['train_loss']) + 1)
    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='train_loss')
    plt.plot(epochs, test_loss, 'r-', label='test_loss')
    plt.title('Ztráta')
    plt.xlabel('Počet epoch')
    #plt.xticks(epochs)
    plt.xticks(range(len(epochs)), [str(epoch) if i % 3 == 0 else '' for i, epoch in enumerate(epochs)])
    plt.grid(True)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b-', label='train_accuracy')
    plt.plot(epochs, test_accuracy, 'r-', label='test_accuracy')
    plt.title('Přesnost')
    plt.xlabel('Počet epoch')
    #plt.xticks(epochs)
    plt.xticks(range(len(epochs)), [str(epoch) if i % 3 == 0 else '' for i, epoch in enumerate(epochs)])
    plt.legend()

    plt.grid(True)
    plt.savefig(f"./plots/plot_{timestamp}_{model_name}_{setup}_acc.png")

    print(f"[INFO] Figurew 'plot_{timestamp}_{model_name}_{setup}_acc.png' saved.")


def plot_textinfo(classes, all_acc, class_acc, precision, recall, f1_score, model_name, setup, timestamp):
    # Sample textual data

    
    plt.figure(figsize=(8, 4))
    plt.title(f'Informace k modelu: {model_name}, epochy: {setup}')

    # Add text to the plot
    plt.text(0.5, 0.8, f"{all_acc=}", ha='center', va='center', fontsize=11)
    plt.text(0.5, 0.7, f"{classes=}", ha='center', va='center', fontsize=11)
    plt.text(0.5, 0.6, f"{class_acc=}", ha='center', va='center', fontsize=11)
    plt.text(0.5, 0.5, f"{precision=}", ha='center', va='center', fontsize=11)
    plt.text(0.5, 0.4, f"{recall=}", ha='center', va='center', fontsize=11)
    plt.text(0.5, 0.3, f"{f1_score=}", ha='center', va='center', fontsize=11)

    # Hide axes
    plt.axis('off')

    plt.savefig(f"./plots/plot_{timestamp}_{model_name}_{setup}_txt.png")

    print(f"[INFO] Figurew 'plot_{timestamp}_{model_name}_{setup}_txt.png' saved.")