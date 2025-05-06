import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou

H = 512
W = 512

def process_image(image_path):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"[LỖI] Không thể đọc ảnh từ: {image_path}")
        return
    
    h, w, _ = original_image.shape
    
    model_input = cv2.resize(original_image, (W, H))
    model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
    model_input = model_input/255.0
    model_input = model_input.astype(np.float32)
    model_input = np.expand_dims(model_input, axis=0)
    
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")
    y = model.predict(model_input)[0]
    y = cv2.resize(y, (w, h))
    y = np.expand_dims(y, axis=-1)
    
    mask = (y > 0.5).astype(np.float32)
    mask = np.repeat(mask, 3, axis=2)
    masked_image = np.zeros_like(original_image)
    masked_image = np.where(mask == 1, original_image, masked_image)
    
    save_dir = "E:/Segmentation/test_images/mask"
    os.makedirs(os.path.join(save_dir, "a"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "b"), exist_ok=True)
    
    image_name = os.path.basename(image_path)
    
    cv2.imwrite(os.path.join(save_dir, "a", image_name), original_image)
    cv2.imwrite(os.path.join(save_dir, "b", image_name), masked_image)

if __name__ == "__main__":
    image_dir = "E:/Segmentation/test_images/image"
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            print(f"Processing: {image_name}")
            process_image(image_path)