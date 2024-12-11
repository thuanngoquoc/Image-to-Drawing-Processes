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
        
        self.sketch_data_folder = "sketch_data"
        if not os.path.exists(self.sketch_data_folder):
            os.makedirs(self.sketch_data_folder, exist_ok=True)

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

        # Nút nhập kích thước
        self.btn_resize = QPushButton("Resize Ảnh")
        self.btn_resize.clicked.connect(self.resize_image_dialog)
        self.btn_resize.setEnabled(False)
        layout.addWidget(self.btn_resize)

        # Nút upload lên Kaggle
        self.btn_upload = QPushButton("Upload lên Kaggle")
        self.btn_upload.clicked.connect(self.upload_to_kaggle)
        self.btn_upload.setEnabled(False)
        layout.addWidget(self.btn_upload)

        # Nút chuyển thành Sketch
        self.btn_to_sketch = QPushButton("Chuyển thành Sketch")
        self.btn_to_sketch.clicked.connect(self.convert_to_sketch)
        self.btn_to_sketch.setEnabled(False)
        layout.addWidget(self.btn_to_sketch)

        # Nút bắt đầu vẽ
        self.btn_start_drawing = QPushButton("Bắt đầu Vẽ")
        self.btn_start_drawing.clicked.connect(self.start_drawing)
        self.btn_start_drawing.setEnabled(False)
        layout.addWidget(self.btn_start_drawing)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.selected_image = None
        self.selected_image_path = None
        self.resized_image = None
        self.sketch_image = None

        # Định nghĩa kích thước nền full screen (ví dụ 1920x1080)
        self.background_w = 1920
        self.background_h = 1080
        # Tạo ảnh nền (ví dụ: nền trắng) hoặc load ảnh nền cố định
        # Ở đây tạm thời tạo nền trắng
        self.background_image = np.ones((self.background_h, self.background_w, 3), dtype=np.uint8)*255

    def choose_image_source(self):
        source = self.source_combo.currentText()
        if "Local" in source:
            self.choose_local_image()
        elif "URL" in source:
            self.choose_url_image()
        elif "Camera" in source:
            self.choose_camera_image()

    def choose_local_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Chọn ảnh', '', 'Image files (*.png *.jpg *.jpeg)')
        if fname:
            img = cv2.imread(fname)
            if img is not None:
                self.selected_image = img
                self.label_info.setText("Đã chọn ảnh từ máy: " + fname)
                self.btn_resize.setEnabled(True)
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
                    self.btn_resize.setEnabled(True)
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
            self.btn_resize.setEnabled(True)
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

    def resize_image_dialog(self):
        if self.selected_image is None:
            QMessageBox.warning(self, "Cảnh báo", "Bạn chưa chọn ảnh!")
            return

        # Yêu cầu người dùng nhập chiều rộng hoặc chiều cao
        # Ví dụ: cho phép nhập chiều rộng, tính chiều cao theo tỷ lệ
        w, h = self.selected_image.shape[1], self.selected_image.shape[0]

        new_width, ok = QInputDialog.getInt(self, "Nhập chiều rộng mới", "Chiều rộng:", value=w, min=10, max=10000)
        if ok:
            ratio = new_width / w
            new_height = int(h * ratio)
            self.resized_image = cv2.resize(self.selected_image, (new_width, new_height))
            # Lưu ảnh resized
            input_image_path = os.path.join(self.data_folder, "input_img.jpg")
            cv2.imwrite(input_image_path, self.resized_image)
            self.selected_image_path = input_image_path
            self.btn_upload.setEnabled(True)
            self.btn_to_sketch.setEnabled(True)
            self.label_info.setText(f"Đã resize ảnh về {new_width}x{new_height}")

    def upload_to_kaggle(self):
        if self.resized_image is None or self.selected_image_path is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có ảnh để upload!")
            return

        print("Đang upload ảnh lên Kaggle dataset...")
        self.data_handler.upload_image("thuanngoquoc/inputdata", self.selected_image_path)
        print("Upload thành công.")

    def convert_to_sketch(self):
        if self.resized_image is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có ảnh resized để chuyển sketch!")
            return

        # Chuyển ảnh resized thành sketch
        sketch = self.image_processor.to_sketch(self.resized_image)

        # Bây giờ đặt ảnh sketch vào giữa nền background
        # Tính toán vị trí để đặt vào giữa
        bg_h, bg_w = self.background_image.shape[:2]
        h, w = sketch.shape[:2]
        start_x = (bg_w - w)//2
        start_y = (bg_h - h)//2

        # Tạo một bản sao nền
        composed = self.background_image.copy()

        # Ở đây, ảnh sketch hiện là nét đen trên nền trắng.  
        # Dán sketch vào nền
        composed[start_y:start_y+h, start_x:start_x+w] = sketch

        # Lưu ảnh này
        sketch_path = os.path.join(self.sketch_data_folder, "final_sketch.jpg")
        cv2.imwrite(sketch_path, sketch)

        self.sketch_image = composed.copy()

        # Hiển thị nó trong DrawingWidget
        self.drawing_widget = DrawingWidget()
        # Ở giai đoạn này chưa set data vì chưa có result.npz
        self.drawing_widget.set_data([], self.sketch_image)
        self.setCentralWidget(self.drawing_widget)

        # Cho phép bắt đầu vẽ sau khi đã chuyển sketch
        self.btn_start_drawing.setEnabled(True)

    def start_drawing(self):
        # Tải result.npz
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

        # Lúc này DrawingWidget đã có self.sketch_image, ta truyền dữ liệu đối tượng vào
        self.drawing_widget.set_data(final_objects, self.sketch_image)
        self.drawing_widget.start_drawing()
