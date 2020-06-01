from generator import Generator

class SuperResolutionNet():
	def __init__(self):
		self.generator = Generator(10)

	def train(self, epochs, batch_size):
		pass