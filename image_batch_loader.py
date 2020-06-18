import os
import numpy as np
import cv2
import helper

class ImageBatchLoader(object):
  def __init__(self, batch_size, directory='data', count=33201, extension='.png'):
    self.directory = f'{os.path.dirname(os.path.abspath(__file__))}/{directory}'
    self.hr_directory = f"{self.directory}/hr"
    self.lr_directory = f"{self.directory}/lr"
    self.count = count
    self.batch_size = batch_size
    self.extension = extension
    self.init_indices = self._read_files()
    self.indices = self.init_indices
    self.flipping = True
  
  def _read_files(self):
    indices = []
    for file in os.listdir(self.hr_directory):
      if file.endswith(self.extension):
        indices.append(int(file[:-4]))
    return indices

  def normalize(self, input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5

  def denormalize(self, input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

  def next_batch(self):
    indices = self._get_random_batch()
    hr_batch = self._load_images(indices, self.hr_directory)
    lr_batch = self._load_images(indices, self.lr_directory)
    return hr_batch, lr_batch

  def reset(self):
    """Reset the indice list"""
    self.indices = np.copy(self.init_indices)

  def _get_random_batch(self):
    batch = []
    for _ in range(0, self.batch_size):
      index = np.random.choice(self.indices, size=1)
      batch.append(index[0])
      self.indices = np.delete(self.indices, index)
    return batch

  def _load_images(self, indices, directory):
    """Read all images with OpenCV and return them normalized"""
    images = []
    for index in indices:
      filename = f"{str(index).zfill(6)}.png"
      full_path = f"{directory}/{filename}"
      image = cv2.imread(full_path)
      images.append(self.normalize(image))
    return np.array(images).astype(np.float32)

  def get_remaining_count(self):
    """Helper method to determine if all files have been seen in an epoch"""
    pass

  def load_images(self, paths):
    images = []
    for path in paths:
      full_path = f"{self.directory}/{path}"
      image = cv2.imread(full_path)
      images.append(self.normalize(image))
    return np.array(images).astype(np.float32)

