from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import helper

class Generator():
  """Wrapper around a keras model for a generator cnn network"""

  def __init__(self, res_blocks=16, upsampling=2):
    self.upsampling = upsampling
    self.res_blocks = res_blocks

    self.generator = self._build_model()

  def get_model(self):
    """Returns the model object"""
    return self.generator

  def _res_block(self, model, filters=64, kernel_size=3, strides=1):
    """Appends a residual block to the sequential model """

    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    return model

  def _upsample_block(self, model, filters=256, kernel_size=3, strides=1):
    """Appends a upsampling 2d block to the sequential model"""

    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = UpSampling2D(size=2)(model)
    model = LeakyReLU(alpha=0.2)(model)
    return model

  def _build_model(self):
    """Create the model"""

    helper.cprint("Building generator network")
    model_input = Input(shape=(None, None, 3))

    # Add pre-residual blocks
    model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(model_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)

     # Add the residual blocks
    for _ in range(self.res_blocks):
      model = self._res_block(model)

    # Add post residuals block
    model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
    #model = BatchNormalization(momentum = 0.5)(model)

    # Add upsampling blocks
    for _ in range(self.upsampling):
      model = self._upsample_block(model)

    model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
    model = Activation('tanh')(model)

    # Create final model oject and name it
    f_model = Model(inputs=model_input, outputs=model)
    return f_model