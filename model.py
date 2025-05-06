
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Add, GlobalAveragePooling2D, Dense, Input, UpSampling2D, Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from ResNet50 import taoResNet50
from ASPP import taoASPP


H = 512
W = 512

def squeeze_and_excitation(input_tensor, ratio=8):
    init = input_tensor
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = init * se
    return x

def create_DeepLabV3Plus(shape):
    input_tensor, skip_connection, features = create_ResNet50(shape)
    
    # ASPP processing
    x_aspp = create_ASPP(features)
    x_aspp = UpSampling2D((4, 4), interpolation="bilinear")(x_aspp)  # 32x32 -> 128x128
    
    # Process second branch
    x_skip = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(skip_connection)
    x_skip = BatchNormalization()(x_skip)
    x_skip = Activation('relu')(x_skip)
    
    # Merge branches (128x128)
    x = Concatenate()([x_aspp, x_skip])
    x = squeeze_and_excitation(x)
    
    # Xử lý và tăng kích thước
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)  # 128x128 -> 256x256
    
    x = Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)  # 256x256 -> 512x512
    
    x = Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(1, 1, padding='same')(x)
    x = Activation("sigmoid")(x)

    model = Model(input_tensor, x)
    return model

if __name__ == "__main__":
    moHinh = taoDeepLabV3Plus((512, 512, 3))
