import numpy as np
import matplotlib.pyplot as plt


def plot_matrix(classes, matrix, model_name, all_acc, class_acc, NUM_EPOCH):
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

    plt.title(f'Confusion Matrix')
    plt.suptitle(f"Vyhodnocení modelu '{model_name}':\nCelková přesnost modelu: {all_acc * 100:.2f} %",
                 x=0.5, y=0.98, fontsize=11, ha='center', va='top')

    # Adjust the subplot to give some space between the plot and the suptitle
    plt.subplots_adjust(top=0.85)
    #plt.colorbar()
    #plt.text(0, 1, "Additional information goes here", horizontalalignment='left', verticalalignment='center', fontsize=10)
    plt.xlabel('Prediḱovaný typ střechy / Přesnost')
    plt.ylabel('Skutečný typ střechy')
    plt.tight_layout()
    # Show the plot
    #plt.show()
    plt.savefig(f"cm_plot_{model_name}.png")
    print(f"[INFO] Figurew 'cm_plot_{model_name}.png' saved.")