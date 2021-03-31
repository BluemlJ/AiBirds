from keras.layers import ReLU, Convolution2D, BatchNormalization, Layer
from keras.initializers import GlorotNormal


class Residual(Layer):
    """The elementary part of a Residual Network."""

    def __init__(self, num_channels, name, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = Convolution2D(num_channels, kernel_size=3, padding='same', strides=strides,
                                   name=name + "_conv1", kernel_initializer=GlorotNormal)
        self.conv2 = Convolution2D(num_channels, kernel_size=3, padding='same',
                                   name=name + "_conv2", kernel_initializer=GlorotNormal)
        self.bn1 = BatchNormalization(name=name + "_bn1")
        self.bn2 = BatchNormalization(name=name + "_bn2")
        self.skip_conv = None
        if use_1x1conv:
            self.skip_conv = Convolution2D(num_channels, kernel_size=1, strides=strides,
                                           name=name + "_skip_conv", kernel_initializer=GlorotNormal)
        else:
            self.skip_conv = Layer()

    def call(self, inputs, **kwargs):
        x = ReLU()(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        skip = self.skip_conv(inputs)
        x += skip
        return ReLU()(x)


class ResNetBlock(Layer):
    """Sequence of residuals."""

    def __init__(self, num_channels, num_residuals, name, first_block=False, **kwargs):
        """Constructor

        :param num_channels: int
        :param num_residuals: number of residuals in this block
        :param name:
        :param first_block:
        :param kwargs:
        """

        super().__init__(name=name, **kwargs)

        self.residuals = []

        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residuals.append(Residual(num_channels, use_1x1conv=True,  # strides=2,
                                               name=name + "_res%d" % i))
            else:
                self.residuals.append(Residual(num_channels, name=name + "_res%d" % i))

    def call(self, x, **kwargs):
        for residual in self.residuals:
            x = residual(x)
        return x
