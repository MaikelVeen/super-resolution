from sr_gan import SRGAN
from dataset_generator import DatasetGenerator
import helper
from colorama import init

if __name__ == "__main__":
    init()
    sr_gan = SRGAN()
    sr_gan.compile()
    sr_gan.train()
