from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import helper


class Discriminator():
	def __init__(self, input_shape, res_blocks=16, upsampling=2):
		self.input_shape = input_shape
		self.upsampling = upsampling
		self.res_blocks = res_blocks
		self.discriminator = self.build_model()

	def get(self):
		return self.discriminator

	def add_dis_block(self, model, filters=64, kernel_size=3, strides=1):
		""" Adds a residual block to the sequential model """
		model.add(Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same"))
		model.add(BatchNormalization(momentum = 0.5))
		model.add(LeakyReLU(alpha=0.2))

	def build_model(self):
		helper.cprint("Building discrimninator model")
		model = Sequential()
		model.add(Input(shape=self.input_shape))

		# Add pre-residual blocks
		model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")) 
		model.add(LeakyReLU(alpha=0.2))

		self.add_dis_block(model, 64, 3, 2)
		self.add_dis_block(model, 128, 3, 1)
		self.add_dis_block(model, 128, 3, 2)
		self.add_dis_block(model, 256, 3, 1)
		self.add_dis_block(model, 256, 3, 2)
		self.add_dis_block(model, 512, 3, 1)
		self.add_dis_block(model, 512, 3, 2)

		model.add(Flatten())
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))

		model.add(Dense(1, activation='sigmoid'))
		return model

