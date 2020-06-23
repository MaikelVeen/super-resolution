from colorama import init
from sr_gan import SRGAN
from dataset_generator import DatasetGenerator

if __name__ == "__main__":
    init()
    sr_gan = SRGAN()
    sr_gan.compile()

    if sr_gan.config['generate_data'] is True:
        dataset_generator = DatasetGenerator('data')

    if sr_gan.config['model_loading']['active'] is True:
        sr_gan.load_weights(sr_gan.config['model_loading']['epoch'])
        sr_gan.train(sr_gan.config['model_loading']['epoch'])
    else:
        sr_gan.train()
