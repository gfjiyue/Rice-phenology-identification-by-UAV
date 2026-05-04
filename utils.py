import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def save_log(file_path, epoch, train_loss, val_loss, acc):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'TrainLoss', 'ValLoss', 'Accuracy'])
        writer.writerow([epoch, train_loss, val_loss, acc])

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true' if normalize else None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()