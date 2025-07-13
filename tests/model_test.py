import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cupy as cp
import numpy as np
from layers.dense import Dense
from layers.conv2d import Conv2D
from layers.ReLU import ReLU
from layers.utils import im2col, col2im, AdamOptimizer, Sequential, compute_accuracy
from layers.softmaxcrossentropyloss import SoftmaxCrossEntropyLoss
from layers.maxpool2d import MaxPool2D
from layers.flatten import Flatten
from layers.dropout import Dropout
from data.data_loader import Data
import time
import json

all_train_losses = {}
all_val_losses = {}
all_train_accuracies = {}
all_val_accuracies = {}
training_times = {}
confusion_matrices = {}

batch_sizes = [32, 64, 128, 256] # Batch sizes to test

for batch_size in batch_sizes:
    # load data
    print(f"[{batch_size}] Loading data...")
    train_loader = Data(path="D:/studia/SieciNeuronowe/dataset/train", batch_size=64, use_cupy=True)
    val_loader = Data(path="D:/studia/SieciNeuronowe/dataset/test", batch_size=64, use_cupy=True)
    print("Data loaded successfully.")
    # MODEL
    model = Sequential([
        Conv2D(input_channels=1, output_channels=8, kernel_size=2, stride=1, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Dropout(0.3),  # Dropout layer with 40% dropout rate
        Flatten(),
        Dense(input_size=8 * 14 * 14, output_size=10)
    ])
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = AdamOptimizer(learning_rate=0.005)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epochs_done_all = {}
    
    best_val_loss = float('inf')
    patience = 5 
    epochs_done = 0
    num_epochs = 50
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"[Batch size: {batch_size}] Epoch {epoch+1}/{num_epochs}")
        train_loss = 0
        train_acc = 0
        num_batches = 0

        for x_batch, y_batch in train_loader:
            logits = model.forward(x_batch)
            loss = loss_fn.forward(logits, y_batch)
            grad = loss_fn.backward()
            model.backward(grad)
            model.update(optimizer)

            train_loss += float(loss)
            train_acc += float(compute_accuracy(logits, y_batch))
            num_batches += 1

        train_loss /= num_batches
        train_acc /= num_batches

        
        val_logits = model.forward(val_loader.X)
        val_loss = loss_fn.forward(val_logits, val_loader.y)
        val_acc = compute_accuracy(val_logits, val_loader.y)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break
        print(f"Loss (train): {train_loss:.4f}, Acc (train): {train_acc:.4f}, "
        f"Loss (val): {float(val_loss):.4f}, Acc (val): {float(val_acc):.4f}, "
        f"Epoch completed in {time.time() - epoch_start_time:.2f} seconds.")
        epochs_done += 1
        train_losses.append(train_loss)
        val_losses.append(float(val_loss))
        train_accuracies.append(train_acc)
        val_accuracies.append(float(val_acc))
    print(f"[Batch size: {batch_size}] Training completed in {time.time() - start_time:.2f} seconds.")
    end_time = time.time()
    training_time = end_time - start_time
    epochs_done_all[batch_size] = epochs_done
    training_times[batch_size] = training_time
    all_train_losses[batch_size] = train_losses
    all_val_losses[batch_size] = val_losses
    all_train_accuracies[batch_size] = train_accuracies
    all_val_accuracies[batch_size] = val_accuracies
    true_labels = val_loader.y.get()
    val_logits = model.forward(val_loader.X)
    val_preds = np.argmax(val_logits.get(), axis=1)

    cm = confusion_matrix(true_labels, val_preds)
    confusion_matrices[batch_size] = cm  
    results = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_accuracy": train_accuracies,
    "val_accuracy": val_accuracies,
    "training_time_seconds": training_time,
    "epochs_done": epochs_done,}

# Save results to JSON file
    with open(F'[{batch_size}]custom_cnn_results.json', 'w') as f:
        json.dump(results, f)

    print("Wyniki zapisane do custom_cnn_results.json")

# plots
epochs = np.arange(1, num_epochs + 1)

plt.figure(figsize=(14, 6))

# Loss comparison
plt.subplot(1, 2, 1)
for batch_size in batch_sizes:
    plt.plot(all_val_losses[batch_size], label=f"Val Loss (BS={batch_size})")
plt.title("Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy comparison
plt.subplot(1, 2, 2)
for batch_size in batch_sizes:
    plt.plot(all_val_accuracies[batch_size], label=f"Val Acc (BS={batch_size})")
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, bs in enumerate(batch_sizes):
    cm = confusion_matrices[bs]
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Batch Size {bs}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.suptitle("Confusion Matrices for Different Batch Sizes", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# Time comparison
plt.figure(figsize=(6, 4))
batch_sizes = list(training_times.keys())
times = list(training_times.values())
epochs_done = list(epochs_done_all.values())
res = times/epochs_done
plt.bar(batch_sizes, times, color='skyblue')
plt.xlabel("Batch Size")
plt.ylabel("Training Time (s)")
plt.title("Training Time per Batch Size")
plt.grid(True, axis='y')
plt.show()