from matplotlib import pyplot as plt
import cv2
import numpy as np
import curses
import os


SAVE_SETTINGS =  {
            'title': 'Super Resolution',
            'tags': ['LR', 'SR', 'HR'],
            'text': {
              'font_color': (255, 255, 255),
              'border_color': (0, 0, 0),
              'font_size': 0.7,
              'font_thickness': 2,
              'border_thickness': 3,
            }
          }

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
	ax.set_xlabel("Epochs")
	ax.set_ylabel("Loss")
	ax.legend()
	ax.set_title('HR & LR losses')
	ax.grid(True)

	ax = fig.add_subplot(1, 2, 2)
	for i in range(len(gan_losses[0])):
		ax.plot([pt[i] for pt in gan_losses], label="GAN loss %s"%i )
	ax.legend()
	ax.set_xlabel("Epochs")
	ax.set_ylabel("Loss")
	ax.set_title('Gan losses')
	ax.grid(True)


def print_progress_bar(stdscr, batch, batch_count, epoch, epochs_count, loss_real, loss_fake, loss_gan, production=False):
	progress = int(50 * batch // batch_count)
	total_progress = int(50 * epoch // epochs_count)

	# Progress bar for all epochs
	stdscr.addstr(1, 1, "Total progress   : ")
	stdscr.addstr("█" * total_progress, curses.color_pair(1))
	stdscr.addstr('-' * (50 - total_progress))

	# Progress bar for current epoch
	stdscr.addstr(2, 1, "Epoch " + str(epoch) + " progress : ")
	stdscr.addstr("█" * progress, curses.color_pair(1))
	stdscr.addstr('-' * (50 - progress))

	stdscr.addstr(5, 1, "LOSS HR  : " + str(loss_real))
	stdscr.addstr(6, 1, "LOSS LR  : " + str(loss_fake))
	stdscr.addstr(7, 1, "LOSS GAN : " + str(loss_gan))

	# A production run will be faster, but won't allow you to terminate the process without
	# shutting down the terminmal.
	if not production:
		stdscr.addstr(8, 1, "Press 'q' to quit training.")
		if stdscr.getch() == ord('q'):
			exit()

	stdscr.refresh()


def save_result(filename, images, settings=SAVE_SETTINGS, axis=1):
	path = f"{os.path.dirname(os.path.abspath(__file__))}/result/{filename}"
	image = np.concatenate(images, axis=axis)

	# Settings
	font = cv2.FONT_HERSHEY_SIMPLEX
	title = settings['title']
	tags = settings['tags']
	font_color = settings['text']['font_color']
	border_color = settings['text']['border_color']
	font_size = settings['text']['font_size']
	font_thickness = settings['text']['font_thickness']
	border_thickness = settings['text']['border_thickness']
	height, width, channels = image.shape

	for i in range(len(tags)):
		text = tags[i]
		x = int(((width / 3) * i) + 10)
		y = int(height * 0.95)
		# Draw border
		cv2.putText(image, text, (x, y), font, font_size, border_color, border_thickness, cv2.LINE_AA)
		# Draw text
		cv2.putText(image, text, (x, y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

	cv2.imwrite(path, image)

