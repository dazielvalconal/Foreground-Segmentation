import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import (
    HorizontalFlip,
    ChannelShuffle,
    CoarseDropout,
    Rotate,
    Compose,
)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path, split=0.1):
    """Load đường dẫn ảnh và mask, chia train/test"""
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    print(f"Tìm thấy {len(X)} ảnh và {len(Y)} mask trong {path}")
    if len(X) == 0 or len(Y) == 0:
        raise FileNotFoundError("Không tìm thấy ảnh hoặc mask trong thư mục.")

    split_size = max(int(len(X) * split), 1)
    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y)


def apply_augmentations(x, y):
    """Trả về danh sách ảnh và mask đã augment"""
    # Kích thước chuẩn
    H, W = 512, 512
    x = cv2.resize(x, (W, H))
    y = cv2.resize(y, (W, H))

    augmented_images = [x]
    augmented_masks = [y]

    # flip
    aug = HorizontalFlip(p=1.0)
    result = aug(image=x, mask=y)
    augmented_images.append(result["image"])
    augmented_masks.append(result["mask"])

    # Gray scale
    x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x_gray = cv2.cvtColor(x_gray, cv2.COLOR_GRAY2BGR)
    augmented_images.append(x_gray)
    augmented_masks.append(y)

    # Channel shuffle
    aug = ChannelShuffle(p=1.0)
    result = aug(image=x, mask=y)
    augmented_images.append(result["image"])
    augmented_masks.append(result["mask"])

    # Coarse dropout
    aug = CoarseDropout(
        p=1.0, max_holes=8, max_height=32, max_width=32, min_holes=1, fill_value=0
    )
    result = aug(image=x, mask=y)
    augmented_images.append(result["image"])
    augmented_masks.append(result["mask"])

    # Rotate
    aug = Rotate(limit=45, p=1.0)
    result = aug(image=x, mask=y)
    augmented_images.append(result["image"])
    augmented_masks.append(result["mask"])

    return augmented_images, augmented_masks


def augment_data(images, masks, save_path, augment=True):
    for x_path, y_path in tqdm(zip(images, masks), total=len(images)):
        name = os.path.splitext(os.path.basename(x_path))[0]

        x = cv2.imread(x_path, cv2.IMREAD_COLOR)
        y = cv2.imread(y_path, cv2.IMREAD_COLOR)

        if x is None or y is None:
            print(f"Lỗi đọc ảnh hoặc mask: {name}")
            continue

        if augment:
            imgs, msks = apply_augmentations(x, y)
        else:
            x = cv2.resize(x, (512, 512))
            y = cv2.resize(y, (512, 512))
            imgs, msks = [x], [y]

        for i, (img, msk) in enumerate(zip(imgs, msks)):
            img_path = os.path.join(save_path, "image", f"{name}_{i}.png")
            mask_path = os.path.join(save_path, "mask", f"{name}_{i}.png")

            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)

            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, msk)


def main():
    np.random.seed(42)
    data_path = "people_segmentation"

    if not os.path.exists(data_path):
        print(f" Không tìm thấy thư mục {data_path}")
        exit(1)

    if not os.path.exists(os.path.join(data_path, "images")) or not os.path.exists(
        os.path.join(data_path, "masks")
    ):
        print(f" Thư mục {data_path} cần có /images và /masks")
        exit(1)

    try:
        (train_x, train_y), (test_x, test_y) = load_data(data_path)
        print(f"Train:\t{len(train_x)} ảnh")
        print(f"Test:\t{len(test_x)} ảnh")

        create_dir("new_data/train/image/")
        create_dir("new_data/train/mask/")
        create_dir("new_data/test/image/")
        create_dir("new_data/test/mask/")

        print(" Đang tăng cường ảnh train...")
        augment_data(train_x, train_y, "new_data/train/", augment=True)

        print(" Đang xử lý ảnh test (không tăng cường)...")
        augment_data(test_x, test_y, "new_data/test/", augment=False)

        print(" Hoàn tất tạo dữ liệu!")

    except Exception as e:
        print(f"⚠ Lỗi xảy ra: {str(e)}")


if __name__ == "__main__":
    main()
