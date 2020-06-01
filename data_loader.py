import os
import cv2
import helper

class DataLoader():
	def __init__(self, directory, extension='.png'):
		dirpath = os.path.dirname(os.path.abspath(__file__))
		self.dir = f'{dirpath}/{directory}'
		self.extension = extension

	def load_images(self):
		pass

	def get_training_set(self):
		pass

	def get_test_set(self):
		pass

	def generate_hr_set(self):
		hr_dir = f"{self.dir}/hr"
		for file in os.listdir(self.dir): 
			if file.endswith(self.extension):
				# Read image
				image = cv2.imread(f"{self.dir}/{file}")

				# Resize image
				image = cv2.resize(image,(256,256), interpolation = cv2.INTER_CUBIC)

				# Write image to file system
				cv2.imwrite(f"{hr_dir}/{file}", image)

	def generate_lr_set(self):
		hr_dir = f"{self.dir}/hr"

		for file in os.listdir(hr_dir): 
			if file.endswith(self.extension):
				# Read image
				image = cv2.imread(f"{hr_dir}/{file}")

				# Resize image
				image = cv2.resize(image,(64,64), interpolation = cv2.INTER_CUBIC)

				# Write image to file system
				cv2.imwrite(f"{self.dir}/lr/{file}", image)