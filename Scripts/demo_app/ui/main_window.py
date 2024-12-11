from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QWidget, 
                             QVBoxLayout, QPushButton, QLabel, QComboBox, QInputDialog)
import cv2
import numpy as np
import os
import requests

from data_io.kaggle_data_handler import KaggleDataHandler
from segmentation.segment_processor import KaggleSegmentProcessor
from utils.priority_handler import PriorityHandler
from utils.image_processor import ImageProcessor
from ui.drawing_widget import DrawingWidget


class MainWindow(QMainWindow):
    def __init__(self, priority_map=None, dataset_slug="thuanngoquoc/inputdata"):
        super().__init__()
        self.setWindowTitle("App Chọn Ảnh và Vẽ")

        if priority_map is None:
            priority_map = {
                "dog": ["head", "ear", "eye", "mouth", "body", "front_leg", "hind_leg"],
                "cat": ["head", "ear", "eye", "mouth", "body", "front_leg", "hind_leg"],
                "car": ["frame", "wheel", "window", "light"],
                "teddy bear": ["head", "ear", "eye", "nose", "mouth", "body", "hand", "foot"]
            }
        
        self.data_handler = KaggleDataHandler()
        self.priority_handler = PriorityHandler(priority_map)
        self.image_processor = ImageProcessor()
        self.segment_processor = KaggleSegmentProcessor(self.data_handler, dataset_slug)

        # Thay vì:
        # self.data_folder = os.path.join(os.getcwd(), "data_folder")
        # Sửa thành:
        self.data_folder = "data_folder"
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            print(f"Đã tạo thư mục: {self.data_folder}")
        else:
            print(f"Đã tồn tại thư mục: {self.data_folder}")
        
        metadata_path = os.path.join(self.data_folder, "dataset-metadata.json")
        if not os.path.exists(metadata_path):
            print("Cảnh báo: Chưa có dataset-metadata.json trong data_folder!")
        else:
            print("Đã tìm thấy dataset-metadata.json trong data_folder.")
        

        # Giao diện chính
        central_widget = QWidget()
        layout = QVBoxLayout()

        self.label_info = QLabel("Chọn nguồn ảnh")
        layout.addWidget(self.label_info)

        self.source_combo = QComboBox()
        self.source_combo.addItem("Local (máy tính)")
        self.source_combo.addItem("URL (đường link Internet)")
        self.source_combo.addItem("Camera (Webcam)")
        layout.addWidget(self.source_combo)

        self.btn_choose_image = QPushButton("Load Ảnh")
        self.btn_choose_image.clicked.connect(self.choose_image_source)
        layout.addWidget(self.btn_choose_image)

        self.btn_process = QPushButton("Xử lý (Upload & Segment)")
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setEnabled(False)
        layout.addWidget(self.btn_process)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.selected_image = None
        self.selected_image_path = None

    def choose_image_source(self):
        source = self.source_combo.currentText()
        if "Local" in source:
            self.choose_local_image()
        elif "URL" in source:
            self.choose_url_image()
        elif "Camera" in source:
            self.choose_camera_image()

    def save_input_image(self):
        input_image_path = os.path.join(self.data_folder, "input_img.jpg")
        success = cv2.imwrite(input_image_path, self.selected_image)
        if success:
            print(f"Đã lưu ảnh vào: {input_image_path}")
            self.selected_image_path = input_image_path
            self.btn_process.setEnabled(True)
        else:
            QMessageBox.warning(self, "Lỗi", "Không lưu được ảnh vào data_folder.")

    def choose_local_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Chọn ảnh', '', 'Image files (*.png *.jpg *.jpeg)')
        if fname:
            img = cv2.imread(fname)
            if img is not None:
                self.selected_image = img
                self.label_info.setText("Đã chọn ảnh từ máy: " + fname)
                self.save_input_image()
            else:
                QMessageBox.warning(self, "Lỗi", "Không đọc được ảnh từ máy.")

    def choose_url_image(self):
        url, ok = QInputDialog.getText(self, "Nhập URL ảnh", "URL:")
        if ok and url:
            try:
                img = self.load_image_from_url(url)
                if img is not None:
                    self.selected_image = img
                    self.label_info.setText("Đã chọn ảnh từ URL: " + url)
                    self.save_input_image()
                else:
                    QMessageBox.warning(self, "Lỗi", "Không tải được ảnh từ URL.")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không tải được ảnh từ URL.\n{e}")

    def choose_camera_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Lỗi", "Không mở được camera.")
            return
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.selected_image = frame
            self.label_info.setText("Đã chụp ảnh từ camera")
            self.save_input_image()
        else:
            QMessageBox.warning(self, "Lỗi", "Không chụp được ảnh từ camera.")

    def load_image_from_url(self, url):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            img_data = response.content
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        else:
            return None

    def process_image(self):
        if self.selected_image is None:
            QMessageBox.warning(self, "Cảnh báo", "Bạn chưa chọn ảnh!")
            return

        # Upload ảnh lên Kaggle dataset
        print("Đang upload ảnh lên Kaggle dataset...")
        self.data_handler.upload_image("thuanngoquoc/inputdata", self.selected_image_path)
        print("Upload thành công.")

        result_file = "result.npz"
        if not os.path.exists(result_file):
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy file kết quả segment result.npz!")
            return

        data = np.load(result_file, allow_pickle=True)
        objects_data = data["objects"].item()
        objects = objects_data["objects"]

        class_counts = {}
        for obj in objects:
            c_name = obj["class_name"]
            class_counts[c_name] = class_counts.get(c_name, 0) + 1
            obj["label"] = f"{c_name}{class_counts[c_name]}"

        final_objects = []
        for obj in objects:
            c_name = obj["class_name"]
            parts = obj["parts"].item() if isinstance(obj["parts"], np.ndarray) else obj["parts"]
            sorted_parts = self.priority_handler.sort_parts(c_name, parts)
            final_objects.append((obj["label"], sorted_parts))

        sketch_img = self.image_processor.to_sketch(self.selected_image)

        self.drawing_widget = DrawingWidget()
        self.drawing_widget.set_data(final_objects, sketch_img)
        self.setCentralWidget(self.drawing_widget)
