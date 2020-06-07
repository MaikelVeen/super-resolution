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

  def load_images(self):
    images_hr = []
    images_lr = []

  def get_low_res(self):
    pass

  def get_high_res(self):
    pass