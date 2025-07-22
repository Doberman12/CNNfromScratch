from PIL import Image
import numpy as np
import cupy as cp
import os


class Data:
    """
    function to load images from a directory and return them as a list of numpy/cupy arrays.
    The images are resized to 128x128 and normalized to the range [0, 1]. The labels are extracted from the directory names.
    """

    def __init__(self, path, batch_size=64, use_cupy=True, shuffle=True):
        self.path = path
        self.use_cupy = use_cupy
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = None
        self.X, self.y = self.load_images()

    def __iter__(self):
        self.current = 0
        if self.use_cupy:
            self.indices = cp.arange(len(self.X))
        else:
            self.indices = np.arange(len(self.X))

        if self.shuffle:
            if self.use_cupy:
                self.indices = cp.random.permutation(self.indices)
            else:
                self.indices = np.random.permutation(self.indices)

        return self

    def __next__(self):
        """Get the next batch of data."""
        if self.current >= len(self.X):
            raise StopIteration
        i = self.current
        j = min(i + self.batch_size, len(self.X))
        self.current = j
        idx = self.indices[i:j]
        batch_X = self.X[idx]
        batch_y = self.y[idx]
        return batch_X, batch_y

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def load_from_cache(self, cache_file):
        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}")
            if self.use_cupy:
                with open(cache_file, "rb") as f:
                    data = cp.load(f)
                    return data["X"], data["y"]
            else:
                data = np.load(cache_file)
                return data["X"], data["y"]
        return None

    def process_image(self, file_path):
        try:
            image = Image.open(file_path).convert("L").resize((28, 28))
            image = np.array(image).astype(np.float32) / 255.0
            image = image[..., np.newaxis]
            return cp.asarray(image) if self.use_cupy else image
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None

    def scan_directories(self):
        X = []
        y = []
        for dict_name in os.listdir(self.path):
            dict_path = os.path.join(self.path, dict_name)
            if not os.path.isdir(dict_path):
                continue
            for file_name in os.listdir(dict_path):
                if not file_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                ):
                    continue
                file_path = os.path.join(dict_path, file_name)
                image = self.process_image(file_path)
                if image is not None:
                    X.append(image)
                    y.append(dict_name)
        return X, y

    def load_images(self):
        """Load images from a directory and return them as a X (stack of numpy/cupy arrays) and y (index).
        Args:
            path (str): Path to the directory containing images.
            use_cupy (bool): If True, return images as cupy arrays, else as numpy arrays.
        Returns:
            - X (stack): Stack of images as numpy/cupy arrays.
            - y (array): Array of indices.
        """
        cache_file = os.path.join(
            self.path, f"cached_data_{'cupy' if self.use_cupy else 'numpy'}.npz"
        )

        cached = self.load_from_cache(cache_file)
        if cached:
            return cached

        X, y = self.scan_directories()
        X = cp.stack(X) if self.use_cupy else np.stack(X)
        X = X.transpose(0, 3, 1, 2)

        label_to_index = {label: idx for idx, label in enumerate(set(y))}
        y = [label_to_index[label] for label in y]
        y = cp.asarray(y) if self.use_cupy else np.asarray(y)

        print(f"Caching dataset to {cache_file}")
        if self.use_cupy:
            with open(cache_file, "wb") as f:
                cp.savez(f, X=X, y=y)
        else:
            np.savez(cache_file, X=X, y=y)

        return X, y

    def split(self, train_ratio=0.8):
        """Split the dataset into training and validation sets."""
        total_len = len(self.X)
        split_idx = int(total_len * train_ratio)

        if self.use_cupy:
            indices = cp.random.permutation(total_len)
        else:
            indices = np.random.permutation(total_len)

        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        X_train = self.X[train_idx]
        y_train = self.y[train_idx]
        X_val = self.X[val_idx]
        y_val = self.y[val_idx]

        train_data = Data.__new__(Data)
        val_data = Data.__new__(Data)

        for obj, X, y in [(train_data, X_train, y_train), (val_data, X_val, y_val)]:
            obj.X = X
            obj.y = y
            obj.use_cupy = self.use_cupy
            obj.batch_size = self.batch_size
            obj.shuffle = self.shuffle
            obj.path = self.path

        return train_data, val_data
