import os
import numpy as np
import cv2

class DatasetGenerator():
    """Util class that generates large amounts of hr and lr images from photos"""

    def __init__(self, directory, extension='.png'):
        dirpath = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = f'{dirpath}/{directory}'
        self.variance_threshold = 300
        self.laplacian_variance_threshold = 50
        self.extension = extension

    def generate_hr_set(self, size, source, destination):
        """Generates a hr set on the filesystem based on a set of images

        Arguments:
            size -- the size of the cropped images
            source -- source directory of data set
            destination -- directory where hr images will be stored
        """
        count = 0

        for file in os.listdir(source):
            if file.endswith(self.extension):
                image = cv2.imread(f"{source}/{file}")
                count = self._crop_image(image, size, count, destination)

    def generate_lr_set(self, downscale_factor, source, destination):
        """Generates a lr set on the filesystem based on a set of images

        Arguments:
            downscale_factor -- the downscale factor used
            source -- source directory of hr data set
            destination -- directory where lr images will be stored
        """
        for file in os.listdir(source):
            if file.endswith(self.extension):
                image = cv2.imread(f"{source}/{file}")
                y, x, z = image.shape

                image = cv2.resize(image, (x // downscale_factor, y // downscale_factor), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f"{destination}/{file}", image)

    def _crop_image(self, image, size, count, destination):
        # Crops an image into squares and saves them to the file system
        for y in range(0, image.shape[0], size):
            for x in range(0, image.shape[1], size):
                # Crop and save if square and high enough variance
                cropped_image = image[y:y + size, x:x + size, :]
                if self._check_validity(cropped_image, size):
                    cv2.imwrite(f"{destination}/{self._get_imname(count)}", cropped_image)
                count += 1
        return count

    def _check_validity(self, image, size):
        if not image.shape[0] == size or not image.shape[1] == size:
            return False

        if self._filter_low_variance(image):
            return False

        if self._filter_blurry_image(image):
            return False

        return True

    def _filter_low_variance(self, image):
        # Filters cropped image when every color channel is below threshold
        below_threshold = 0
        for i in range(3):
            if np.var(image[:, :, i]) < self.variance_threshold:
                below_threshold += 1
        return below_threshold == 3

    def _filter_blurry_image(self, image):
        """ Filters blurry images out """
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var < self.laplacian_variance_threshold

    @staticmethod
    def _get_imname(count):
        # Returns the filename of a generated image
        return f"{str(count).zfill(6)}.png"
