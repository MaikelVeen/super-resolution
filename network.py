from generator import Generator
from discriminator import Discriminator


class SuperResolutionNet():
	def __init__(self):
		gen = Generator((64,64,3))
		self.generator = gen.get()

		dis = Discriminator((64,64,3))
		self.discriminator = dis.get()

		self.generator.summary()
		self.discriminator.summary()
		

	def train(self, epochs, batch_size):
		pass

	def test(self):
		pass