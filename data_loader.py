import os
import cv2
import helper
import numpy as np

class DataLoader():
  def __init__(self, directory, extension='.png', generate_data=False, size=224, downscale_factor=4):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    self.data_dir = f'{dirpath}/{directory}'
    self.hr_dir = f"{self.data_dir}/hr"
    self.lr_dir = f"{self.data_dir}/lr"
    self.extension = extension

    if generate_data:
      self._generate_hr_set(size)
      self._generate_lr_set(downscale_factor)


  def load_images(self):
    images_hr = []
    images_lr = []

  def get_low_res(self):
    pass

  def get_high_res(self):
    pass


  def _crop(self, img, cropx, cropy, divisionx, divisiony):
    y, x, z = img.shape
    startx = x // divisionx - (cropx // divisiony)
    starty = y // divisionx - (cropy // divisiony)
    return img[starty:starty + cropy, startx:startx + cropx, :]

  def _get_name(self, count):
    return str(count).zfill(6) + '.png'

  def _generate_hr_set(self, size):
    count = 0

    for file in os.listdir(self.data_dir):
      if file.endswith(self.extension):
        image = cv2.imread(f"{self.data_dir}/{file}")

        for i in range(1, 5 ):
          count = self._crop_image(image, size, i, count)

  def _crop_image(self, image, size, division, count):
    """ Crops to a specific part of the image given by the division """

    count = count + 1
    image_crop = self._crop(image, size, size, division, division)
    cv2.imwrite(f"{self.hr_dir}/{self._get_name(count)}", image_crop)
    return count


  def _generate_lr_set(self, downscale_factor):
    for file in os.listdir(self.hr_dir):
      if file.endswith(self.extension):
        image = cv2.imread(f"{self.hr_dir}/{file}")
        y, x, z = image.shape

        image = cv2.resize(image, (x // downscale_factor, y // downscale_factor), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{self.lr_dir}/{file}", image)
