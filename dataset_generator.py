import os
import numpy as np
import cv2

class DatasetGenerator():
  """Util class that generates large amounts of hr and lr images from photos"""

  def __init__(self, directory, extension='.png', size=224, downscale_factor=4):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    self.data_dir = f'{dirpath}/{directory}'
    self.hr_dir = f"{self.data_dir}/hr"
    self.lr_dir = f"{self.data_dir}/lr"
    self.variance_threshold = 300
    self.extension = extension
    self._generate_hr_set(size)
    self._generate_lr_set(downscale_factor)

  def _get_imname(self, count):
    """Returns the filename of a generated image"""
    return f"{str(count).zfill(6)}.png"

  def _generate_hr_set(self, size):
    count = 0

    for file in os.listdir(self.data_dir):
      if file.endswith(self.extension):
        image = cv2.imread(f"{self.data_dir}/{file}")
        count = self._crop_image(image, size, count)

  def _crop_image(self, image, size, count):
    """Crops an image into squares and saves them to the file system"""
    for y in range(0, image.shape[0], size):
      for x in range(0, image.shape[1], size):
        # Crop and save if square and high enough variance
        cropped_image = image[y:y + size, x:x + size, :]
        if self._check_validity(cropped_image):
          cv2.imwrite(f"{self.hr_dir}/{self._get_imname(count)}", cropped_image)
        count += 1
    return count
  
  def _generate_lr_set(self, downscale_factor):
    for file in os.listdir(self.hr_dir):
      if file.endswith(self.extension):
        image = cv2.imread(f"{self.hr_dir}/{file}")
        y, x, z = image.shape

        image = cv2.resize(image, (x // downscale_factor, y // downscale_factor), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{self.lr_dir}/{file}", image)

  def _check_validity(self, image):
    if not image.shape[0] == image.shape[1]:
      return False

    if not _filter_low_variance(image):
      return False

    return True
  
  def _filter_low_variance(self, image):
    """ Filters cropped image when every color channel is below threshold """
    below_threshold = 0
    for i in range(3):
      if(np.var(image[:, :, i]) < self.variance_threshold):
        below_threshold+=1
    if below_threshold == 3:
      return False
    else:
      return True
