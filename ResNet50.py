import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Add, Input
from tensorflow.keras.models import Model

def basic_block(x, filters, kernel_size=3, stride=1, downsample=False):
    x_skip = x
    
    if downsample:
        x = Conv2D(filters[0], 1, strides=stride, padding='same')(x)
    else:
        x = Conv2D(filters[0], kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters[1], kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters[2], 1, padding='same')(x)
    x = BatchNormalization()(x)
    
    if x_skip.shape[-1] != filters[2]:
        x_skip = Conv2D(filters[2], 1, strides=stride, padding='same')(x_skip)
        x_skip = BatchNormalization()(x_skip)
    
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

def create_ResNet50(input_shape):
    input_tensor = Input(input_shape)
    
    x = Conv2D(64, 7, strides=2, padding='same')(input_tensor) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)  
    
    x = basic_block(x, [64, 64, 256], stride=1, downsample=True)
    x_skip = x  
    x = basic_block(x, [64, 64, 256])
    x = basic_block(x, [64, 64, 256])
    
    x = basic_block(x, [128, 128, 512], stride=2, downsample=True) 
    for _ in range(3):
        x = basic_block(x, [128, 128, 512])
    
    x = basic_block(x, [256, 256, 1024], stride=2, downsample=True) 
    for i in range(5):
        x = basic_block(x, [256, 256, 1024])
    features = x  
    
    return input_tensor, x_skip, features
    
    # Conv5
    x = khoiCoBan(x, [512, 512, 2048], buoc=2, giamChieu=True)
    x = khoiCoBan(x, [512, 512, 2048])
    x = khoiCoBan(x, [512, 512, 2048])
    
    moHinh = Model(dauVao, x)
    return moHinh