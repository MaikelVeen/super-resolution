from sr_gan import SRGAN
from dataset_generator import DatasetGenerator
import helper
from colorama import init

if __name__ == "__main__":
  init()
  helper.gprint("Starting...")
  #data_gen = DatasetGenerator('data')
  sr_gan = SRGAN()
