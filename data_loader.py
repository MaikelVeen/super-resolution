import os
import numpy as np
import cv2
import helper

class DataLoader():
  def __init__(self, directory, extension='.png'):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    self.data_dir = f'{dirpath}/{directory}'
    self.hr_dir = f"{self.data_dir}/hr"
    self.lr_dir = f"{self.data_dir}/lr"
    self.extension = extension

    self._load_images()

  def _load_images(self):
    self.images_hr = self.normalize(self._load_set(self.hr_dir))
    self.images_lr = self.normalize(self._load_set(self.lr_dir))

  def _load_set(self, directory):
    images = []
    for file in os.listdir(directory):
      if file.endswith(self.extension):
        images.append(cv2.imread(f"{directory}/{file}"))
    return np.array(images)

  def get_lr_set(self):
    return self.images_lr

  def get_hr_set(self):
    return self.images_hr

  def normalize(self, input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5

  def denormalize(self, input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
