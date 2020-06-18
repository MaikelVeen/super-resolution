import os
import numpy as np
import cv2


class ImageBatchLoader():
    """Batch loader that returns randomly sampled images batches

    Loader assumes all the images in the hr and lr directories
    are named the same and that the directories are of equal size

    Attributes:
        directory (str): Path of top level data directory
        hr_directory (str): Path of hr image directory
        batch_size (int): Path of lr image directory
        extension (str): Allowed extension for images
        init_indices (arr): List of all the filenames in the hr (and lr) directory
        indices (arr): List of filenames with currently sampled removed
    """

    def __init__(self, batch_size=20, directory='data', extension='.png'):
        self.directory = f'{os.path.dirname(os.path.abspath(__file__))}/{directory}'
        self.hr_directory = f"{self.directory}/hr"
        self.lr_directory = f"{self.directory}/lr"
        self.batch_size = batch_size
        self.extension = extension
        self.init_indices = self._get_indices()
        self.indices = self.init_indices

    def load_images(self, paths):
        """Returns an ndarray with image matrices

        Arguments:
        paths -- an array of relative paths to the data directory
        """
        images = []

        for path in paths:
            full_path = f"{self.directory}/{path}"
            image = cv2.imread(full_path)
            images.append(self.normalize(image))

        return np.array(images).astype(np.float32)

    @staticmethod
    def normalize(input_data):
        """Returns an ndarray with normalized image matrices

        Arguments:
        input_data -- ndarray of image matrices
        """
        return (input_data.astype(np.float32) - 127.5) / 127.5

    @staticmethod
    def denormalize(input_data):
        """Returns an ndarray with denormalized image matrices

        Arguments:
        input_data -- ndarray of normalized image matrices
        """
        input_data = (input_data + 1) * 127.5
        return input_data.astype(np.uint8)

    def next_batch(self):
        """Returns the next random hr and lr image batch"""
        indices = self._get_random_batch()
        hr_batch = self._load_images(indices, self.hr_directory)
        lr_batch = self._load_images(indices, self.lr_directory)
        return hr_batch, lr_batch

    def reset(self):
        """Reset the indice list to the initial list of indices"""
        self.indices = np.copy(self.init_indices)

    def _get_indices(self):
        # Returns a list with the filenames of the images
        indices = []
        for file in os.listdir(self.hr_directory):
            if file.endswith(self.extension):
                indices.append(int(file[:-4]))
        return indices

    def _get_random_batch(self):
        # Randomly sample a list of indices from the pool
        batch = []
        for _ in range(0, self.batch_size):
            index = np.random.choice(self.indices, size=1)
            batch.append(index[0])
            self.indices = np.delete(self.indices, index)
        return batch

    def _load_images(self, indices, directory):
        # Return an ndarray with image matrices
        images = []
        for index in indices:
            full_path = f"{directory}/{str(index).zfill(6)}.png"
            image = cv2.imread(full_path)
            images.append(self.normalize(image))
        return np.array(images).astype(np.float32)
