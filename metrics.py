
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import UpSampling2D

H = 512
W = 512
SMOOTH = 1e-15

def resize_to_match(y_pred, y_true):
    if y_pred.shape[1] != y_true.shape[1] or y_pred.shape[2] != y_true.shape[2]:
        y_pred = UpSampling2D(size=(2, 2), interpolation='bilinear')(y_pred)
    return y_pred

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        y_pred = tf.keras.backend.get_value(resize_to_match(y_pred, y_true))
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_thuc, y_du_doan], tf.float32)

def dice_coef(y_true, y_pred):
    y_pred = resize_to_match(y_pred, y_true)
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true = tf.reshape(y_true, [-1, H*W])
    y_pred = tf.reshape(y_pred, [-1, H*W])
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    return tf.reduce_mean((2. * intersection + SMOOTH) / (tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1) + SMOOTH))

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3]) - intersection
    
    return tf.reduce_mean((intersection + 1e-15) / (union + 1e-15))
