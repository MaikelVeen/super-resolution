from matplotlib import pyplot as plt

def bprint(text):
	""" Prints the text in blue """
	print(f'\033[94m {text} \033[0m')


def gprint(text):
	""" Prints the text in green """
	print(f'\033[92m {text} \033[0m')


def cprint(text):
	""" Prints the text in cyan """
	print(f'\033[36m {text} \033[0m')


def plot_loss(real_losses, fake_losses, gan_losses):
	""" Plots HR, LR and all Gan Losses"""
	fig = plt.figure(figsize=(12, 6))
	ax = fig.add_subplot(1, 2, 1)
	ax.plot(real_losses,'r-x', label="Loss HR")
	ax.plot(fake_losses,'b-x', label="Loss LR")
	ax.xlabel("Epochs")
	ax.ylabel("Loss")
	ax.legend()
	ax.set_title('HR & LR losses')
	ax.grid(True)

	ax = fig.add_subplot(1, 2, 2)
	for i in range(len(gan_losses[0])):
		ax.plot([pt[i] for pt in gan_losses], label="GAN loss %s"%i )
	ax.legend()
	ax.xlabel("Epochs")
	ax.ylabel("Loss")
	ax.set_title('Gan losses')
	ax.grid(True)