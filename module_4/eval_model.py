import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter


def denormalize_inplace(tensor):
    """Denormalize a tensor in place."""
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2).to(tensor.device)
    tensor.mul_(std).add_(mean)
    return tensor


def matrix_and_accuracy(model, test_loader, device, OUT_FEAT, class_names, writer):
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
            print(f"INPUT: {inputs.shape}")
            print(f"LABEL: {labels.shape}")
            print(f"PREDICTED: {predicted.shape}")

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

            # Save mismatched images to TensorBoard
            # mismatched_indices = (predicted != labels).nonzero() # catches only first 10 or so mismatches
            mismatched_indices = []
            for idx, (p, l) in enumerate(zip(predicted, labels)):
                if p != l:
                    mismatched_indices.append(idx)
            #mismatched_indices = torch.tensor(list(set(mismatched_indices)))
            for idx in mismatched_indices:
                image = inputs[idx].to(device)
                true_idx = labels[idx].item()
                true_label = class_names[true_idx]
                pred_idx = predicted[idx].item()
                pred_label = class_names[pred_idx]
                writer.add_image(f"Neshoda/{true_label}_predicted_as_{pred_label}", 
                                 torchvision.utils.make_grid(denormalize_inplace(image)))
    
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