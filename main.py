import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                          QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                          QMessageBox, QFrame, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Thêm imports cần thiết
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou

# Thêm vào phần import
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                           QMessageBox, QFrame, QSpacerItem, QSizePolicy)

class SegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ứng dụng Tách & Thay Nền Ảnh")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color: #f0f0f0;")
        
        # Load model
        try:
            model_path = 'E:\\Segmentation\\files\\model.h5'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
                self.model = load_model(model_path)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Lỗi", f"Không thể load model: {str(e)}")
            sys.exit(1)
            
        # Tạo widget chính
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout chính
        main_layout = QVBoxLayout()
        content_layout = QHBoxLayout()
        
        # Tạo frame cho các phần
        left_frame = QFrame()
        middle_frame = QFrame()
        right_frame = QFrame()
        
        # Style cho frame
        frame_style = """
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
            }
        """
        left_frame.setStyleSheet(frame_style)
        middle_frame.setStyleSheet(frame_style)
        right_frame.setStyleSheet(frame_style)
        
        # Layout cho ảnh gốc
        left_layout = QVBoxLayout(left_frame)
        self.original_label = QLabel("Ảnh Gốc")
        self.original_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_image = QLabel()
        self.original_image.setFixedSize(400, 400)
        self.original_image.setStyleSheet("border: 2px solid #ddd;")
        left_layout.addWidget(self.original_label)
        left_layout.addWidget(self.original_image)
        
        # Layout cho buttons ở giữa
        middle_layout = QVBoxLayout(middle_frame)
        self.load_button = QPushButton("Tải Ảnh")
        self.process_button = QPushButton("Tách Nền")
        self.change_bg_button = QPushButton("Thay Nền")
        self.save_button = QPushButton("Lưu Ảnh")
        
        # Thêm spacer để căn giữa các nút
        middle_layout.addStretch()
        middle_layout.addWidget(self.load_button)
        middle_layout.addSpacing(20)
        middle_layout.addWidget(self.process_button)
        middle_layout.addSpacing(20)
        middle_layout.addWidget(self.change_bg_button)
        middle_layout.addSpacing(20)
        middle_layout.addWidget(self.save_button)
        middle_layout.addStretch()
        
        # Layout cho ảnh kết quả
        right_layout = QVBoxLayout(right_frame)
        self.result_label = QLabel("Ảnh Kết Quả")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_image = QLabel()
        self.result_image.setFixedSize(400, 400)
        self.result_image.setStyleSheet("border: 2px solid #ddd;")
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.result_image)
        
        # Thêm các frame vào layout chính
        content_layout.addWidget(left_frame)
        content_layout.addWidget(middle_frame)
        content_layout.addWidget(right_frame)
        main_layout.addLayout(content_layout)
        
        main_widget.setLayout(main_layout)
        
        # Style cho buttons
        button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 15px;
                font-size: 14px;
                border-radius: 8px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """
        self.load_button.setStyleSheet(button_style)
        self.process_button.setStyleSheet(button_style)
        self.change_bg_button.setStyleSheet(button_style)
        self.save_button.setStyleSheet(button_style)
        
        # Thiết lập trạng thái ban đầu
        self.process_button.setEnabled(False)
        self.change_bg_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # Kết nối signals
        self.load_button.clicked.connect(self.load_image)
        self.process_button.clicked.connect(self.process_image)
        self.change_bg_button.clicked.connect(self.change_background)
        self.save_button.clicked.connect(self.save_image)
        
        # Biến để lưu trữ mask và ảnh đã xử lý
        self.current_mask = None
        self.processed_image = None

    def change_background(self):
        try:
            bg_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Chọn ảnh nền", 
                "", 
                "Image Files (*.png *.jpg *.jpeg)"
            )
            if bg_path:
                # Đọc ảnh nền mới và điều chỉnh kích thước
                new_bg = cv2.imread(bg_path)
                new_bg = cv2.resize(new_bg, (512, 512))
                
                # Đọc lại ảnh gốc
                original = cv2.imread(self.image_path)
                original = cv2.resize(original, (512, 512))
                
                # Chuyển mask thành định dạng phù hợp
                mask = np.uint8(self.current_mask * 255)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                # Tạo mask cho nền
                inv_mask = cv2.bitwise_not(mask)
                
                # Tách đối tượng từ ảnh gốc
                foreground = cv2.bitwise_and(original, mask)
                
                # Tách phần nền mới
                background = cv2.bitwise_and(new_bg, inv_mask)
                
                # Kết hợp đối tượng và nền
                result = cv2.add(foreground, background)
                
                # Chuyển đổi màu và hiển thị
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                height, width, channel = result_rgb.shape
                bytes_per_line = 3 * width
                q_img = QImage(result_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.result_image.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
                self.processed_image = result
                self.save_button.setEnabled(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi thay nền: {str(e)}")

    def save_image(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu ảnh",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg)"
            )
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
                QMessageBox.information(self, "Thành công", "Đã lưu ảnh thành công!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi lưu ảnh: {str(e)}")

    def process_image(self):
        try:
            # Đọc ảnh và tiền xử lý
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
            img = cv2.resize(img, (512, 512))  # Resize thành 512x512
            img = img / 255.0
            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0)
            
            # Dự đoán mask
            pred_mask = self.model.predict(img)
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            pred_mask = pred_mask[0]
            
            # Áp dụng mask vào ảnh gốc
            original = cv2.imread(self.image_path)
            original = cv2.resize(original, (512, 512))
            
            # Tạo mask 3 kênh màu và điều chỉnh kích thước
            mask_3channel = np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2) * 255
            mask_3channel = mask_3channel.astype(np.uint8)
            
            # Áp dụng mask
            result = cv2.bitwise_and(original, original, mask=pred_mask[:, :, 0])
            
            # Chuyển đổi màu để hiển thị
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            # Hiển thị kết quả với kích thước phù hợp
            height, width, channel = result.shape
            bytes_per_line = 3 * width
            q_img = QImage(result.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Tính toán kích thước để hiển thị đầy đủ ảnh trong QLabel
            label_size = self.result_image.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.result_image.setPixmap(scaled_pixmap)
            
            # Lưu mask và ảnh đã xử lý
            self.current_mask = pred_mask[:, :, 0]
            self.processed_image = result
            
            # Kích hoạt nút thay nền
            self.change_bg_button.setEnabled(True)
            self.save_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Chọn ảnh", 
            "", 
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            try:
                self.image_path = file_name
                pixmap = QPixmap(file_name)
                scaled_pixmap = pixmap.scaled(512, 512, Qt.KeepAspectRatio)
                self.original_image.setPixmap(scaled_pixmap)
                self.process_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể tải ảnh: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec_())