import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Concatenate, UpSampling2D

def create_ASPP(input_tensor):
    shape = input_tensor.shape
    
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(input_tensor)
    y1 = Conv2D(256, 1, padding="same", use_bias=False)(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

    y2 = Conv2D(256, 1, padding="same", use_bias=False)(input_tensor)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(input_tensor)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(input_tensor)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(input_tensor)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(256, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y