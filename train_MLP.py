from architecture import MLP
from DataLoader import FeatureDataset
import matplotlib.pyplot as plt
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader


def main():
    # Open JSON file with name of training and test files
    with open('training_files.json', 'r') as f:
        json_string = f.read()
        names = json.loads(json_string)

    train_file_paths = np.array(names['train_files'])
    test_file_paths = np.array(names['test_files'])

    for i in range(len(train_file_paths)):
        train_dataset = FeatureDataset(train_file_paths[i])
        train_loader = DataLoader(train_dataset, batch_size=600, shuffle=True)

    for i in range(len(test_file_paths)):
        test_dataset = FeatureDataset(test_file_paths[i])
        test_loader = DataLoader(test_dataset, batch_size=250, shuffle=True)

    model = MLP(6, 2)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Train the model
    # Initialize lists to store the accuracy values
    train_acc = []
    test_acc = []
    # Track the losses at each epoch
    losses = []
    test_losses = []
    num_epochs = 200
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(features)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate the accuracy of the model on the train set
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            acc = correct / total

        train_acc.append(acc)

        # Save the loss value at this epoch
        losses.append(loss.item())

        if (epoch + 1) % 2 == 0:
            # Print the loss value at the end of each epoch
            print(f'Epoch [{epoch + 1}/200], Loss: {loss.item():.4f}')
            print(f'Accuracy: {train_acc[-1]:.2f}%')

        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            test_probs = []  # list to store the predicted probabilities
            test_labels = []  # list to store the true labels
            for i, (feature, label) in enumerate(test_loader):
                outputs = model(feature)
                test_loss = loss_fn(outputs, label)
                test_probs.append(outputs)
                test_labels.append(label.detach().numpy())
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                test_accuracy = correct / total

        test_losses.append(test_loss.item())
        test_acc.append(test_accuracy)

    y_probs = np.concatenate(test_probs)
    y_true = np.concatenate(test_labels)

    fpr = np.array([0])
    tpr = np.array([0])
    valid_at = np.array([0])
    for i in range(1, 1001):
        # Calculate the AUC-ROC score
        threshold = i * 0.001
        y_probs_n = y_probs.copy()
        y_probs_n = y_probs_n[:, 1]
        # Extract values that are above the threshold and set to 1
        y_probs_n = np.where(y_probs_n > threshold, 1, 0)

        # Calculate ROC curve
        fpr_n, tpr_n, thresholds_n = roc_curve(y_true, y_probs_n)
        if fpr_n.size < 3:
            continue
        valid_at = np.append(valid_at, i)
        fpr = np.append(fpr, fpr_n[1])
        tpr = np.append(tpr, tpr_n[1])

    sorted_fpr = np.argsort(fpr)
    sorted_tpr = np.argsort(tpr)
    fpr = fpr[sorted_fpr]
    tpr = tpr[sorted_tpr]
    valid_at = valid_at[sorted_fpr]
    valid_at = valid_at * .001
    valid_at = np.round(valid_at, 1)

    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # # for i in range(fpr.shape[0]):
    # #     v = np.array2string(valid_at[i])
    # #     plt.text(fpr[i], tpr[i], v, color="red", fontsize=12)
    # plt.xlim([-0.05, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('AUC-ROC Curve')
    # plt.legend(loc="lower right")
    # plt.gcf().set_facecolor('white')
    # # plt.savefig('AUC-ROC Curve.png')
    # plt.show()
    # Plot the loss curve
    plt.plot(losses, label='Train')
    plt.plot(test_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.gcf().set_facecolor('white')
    # plt.savefig('loss_curve.png')
    plt.show()

    # Plot the accuracy curve
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.gcf().set_facecolor('white')
    # plt.savefig('accuracy_curve.png')
    plt.show()

    # Save Model
    name = "MLP.pt"
    torch.save(model, name)




if __name__ == '__main__':
    main()
