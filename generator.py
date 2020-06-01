from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import helper

class Generator():
	def __init__(self, noise_shape, res_blocks=16, upsampling=4):
		self.noise_shape = noise_shape
		self.upsampling = upsampling
		self.res_blocks = res_blocks
		self.generator = self.build_model()

	def get(self):
		return self.generator

	def add_res_block(self, model, filters=64, kernel_size=3, strides=1):
		""" Adds a residual block to the sequential model """
		model.add(Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same"))
		model.add(BatchNormalization(momentum = 0.5))
		model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2]))
		model.add(Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same"))
		model.add(BatchNormalization(momentum = 0.5))

	def add_deconvolution(self, model, filters=256, kernel_size=3, strides=1):
		"""" Adds a deconvolution 2d block to the sequential model"""
		model.add(Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same"))
		model.add(UpSampling2D(size = 2))
		model.add(LeakyReLU(alpha = 0.2))

	def build_model(self):
		helper.cprint("Building generator model")
		model = Sequential()
		model.add(Input(shape=self.noise_shape))

		# Add pre-residual blocks
		model.add(Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")) 
		model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2]))

		# Add the residual blocks
		for i in range(self.res_blocks):
			self.add_res_block(model)

		# Add post residuals block
		model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same"))
		model.add(BatchNormalization(momentum = 0.5))

		# Add upsampling blocks
		for j in range(self.upsampling):
			self.add_deconvolution(model)

		model.add(Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same"))
		model.add(Activation('tanh'))
		return model

