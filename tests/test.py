from data.data_loader import load_images_from_directory
import cupy as cp
loaded_images, labels = load_images_from_directory('data/test_images', use_cupy=False)

def conv(x,w,b):
    product = x * w
    output = cp.sum(product, axis=(0, 1)) + b  # Convolution operation
    return output

x = loaded_images[1]
w = cp.random.randn(3, 3, 3, 16) * 0.01  # Example kernel
b = cp.zeros(16)  # Example bias
output = conv(x, w, b)
print("Output shape:", output.shape)
print("Output:", output)

