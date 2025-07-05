import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json

with open('./[32]custom_cnn_results.json', 'r') as f:
    S32 = json.load(f)
with open('./[64]custom_cnn_results.json', 'r') as f:
    S64 = json.load(f)
with open('./[128]custom_cnn_results.json', 'r') as f:
    CuPy128 = json.load(f)
with open('./[256]custom_cnn_results.json', 'r') as f:
    S256 = json.load(f) 
with open('./TF_results.json', 'r') as f:
    TensorFlow128 = json.load(f)
print("Data loaded successfully.")
# plt.subplot(1, 2, 1)
# for D in comp:
#     if D == CuPy128:
#         label = "CuPy 128"
#     elif D == TensorFlow128:
#         label = "TensorFlow 128"
#     plt.plot(D['val_loss'], label=f"Val Loss ({label})")
# plt.title("Validation Loss per Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)

# # Accuracy comparison
# plt.subplot(1, 2, 2)
# for D in comp:
#     if D == CuPy128:
#         label = "CuPy 128"
#     elif D == TensorFlow128:
#         label = "TensorFlow 128"
#     plt.plot(D['val_accuracy'], label=f"Val Accuracy ({label})")
# plt.title("Validation Accuracy per Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

comp = [S32, S64, CuPy128, S256, TensorFlow128]
x = ['CuPy32', 'CuPy64', 'CuPy128', 'CuPy256', 'TensorFlow128']
Y = []

for D in comp:
    avg_time = D['training_time_seconds'] / D['epochs_done']
    Y.append(avg_time)
    print("done")
sns.barplot(x = x, y = Y)
plt.grid(axis='y')
plt.title("Average Training Time per Epoch [seconds]")
plt.show()