import torch
import numpy as np

def matrix_and_accuracy(model, test_loader, device, OUT_FEAT):
    # Initialize confusion matrix
    num_classes = OUT_FEAT
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Initialize variables for calculating overall accuracy
    total_samples = 0
    correct_predictions = 0

    # Initialize variables for calculating class-wise accuracy
    class_correct = torch.zeros(num_classes, dtype=torch.int64)
    class_total = torch.zeros(num_classes, dtype=torch.int64)

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                cm[t.long(), p.long()] += 1

            # Update overall accuracy
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Update class-wise accuracy
            for i in range(num_classes):
                class_correct[i] += ((predicted == i) & (labels == i)).sum().item()
                class_total[i] += (labels == i).sum().item()
    
    # convert torch.tensor to np.array
    cm_np = cm.numpy()
    precision = np.diag(cm_np) / np.sum(cm_np, axis=0)
    recall = np.diag(cm_np) / np.sum(cm_np, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculate overall accuracy
    all_acc = correct_predictions / total_samples

    # Calculate class-wise accuracy
    class_acc = class_correct / class_total

    return cm, cm_np, all_acc, class_acc, precision, recall, f1_score