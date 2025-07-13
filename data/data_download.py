import os
from PIL import Image
from torchvision import datasets


def save_mnist_images(data, labels, output_dir):
    """
    function to save MNIST images as PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(10):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

    for idx, (image, label) in enumerate(zip(data, labels)):
        label = int(label)
        image = Image.fromarray(image.numpy(), mode="L")
        image.save(os.path.join(output_dir, str(label), f"{idx:05d}.png"))


# download MNIST dataset
train_dataset = datasets.MNIST(root="./mnist_data", train=True, download=True)
test_dataset = datasets.MNIST(root="./mnist_data", train=False, download=True)

# save images
save_mnist_images(train_dataset.data, train_dataset.targets, "./dataset/train")
save_mnist_images(test_dataset.data, test_dataset.targets, "./dataset/test")

print("image saved as PNG.")
