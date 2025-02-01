from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QWidget, 
                             QVBoxLayout, QPushButton, QLabel, QComboBox, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

import cv2
import numpy as np
import os
import requests

from data_io.kaggle_data_handler import KaggleDataHandler
from segmentation.segment_processor import KaggleSegmentProcessor
from utils.priority_handler import PriorityHandler
from utils.image_processor import ImageProcessor

class MainWindow(QMainWindow):
    def __init__(self, priority_map=None, dataset_slug="thuanngoquoc/inputdata"):
        super().__init__()
        self.setWindowTitle("Image Selection and Drawing App")

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

        self.sketch_data_folder = "sketch_data"
        if not os.path.exists(self.sketch_data_folder):
            os.makedirs(self.sketch_data_folder, exist_ok=True)

        self.background_image = cv2.imread("background.jpg")
        if self.background_image is None:
            self.background_image = np.ones((1080, 1920, 3), dtype=np.uint8)*255
        else:
            self.background_image = cv2.resize(self.background_image, (1920, 1080))

        central_widget = QWidget()
        self.layout = QVBoxLayout()

        self.label_info = QLabel("Select image")
        self.layout.addWidget(self.label_info)

        self.source_combo = QComboBox()
        self.source_combo.addItem("Local")
        self.source_combo.addItem("URL")
        self.source_combo.addItem("Camera")
        self.layout.addWidget(self.source_combo)

        self.btn_choose_image = QPushButton("Load image")
        self.btn_choose_image.clicked.connect(self.choose_image_source)
        self.layout.addWidget(self.btn_choose_image)

        self.btn_resize = QPushButton("Resize image")
        self.btn_resize.clicked.connect(self.resize_image_dialog)
        self.btn_resize.setEnabled(False)
        self.layout.addWidget(self.btn_resize)

        self.btn_upload = QPushButton("Upload to Kaggle")
        self.btn_upload.clicked.connect(self.upload_to_kaggle)
        self.btn_upload.setEnabled(False)
        self.layout.addWidget(self.btn_upload)

        self.btn_to_sketch = QPushButton("Convert to Sketch")
        self.btn_to_sketch.clicked.connect(self.convert_to_sketch)
        self.btn_to_sketch.setEnabled(False)
        self.layout.addWidget(self.btn_to_sketch)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.image_label.setVisible(False)

        self.btn_start_drawing = QPushButton("Start Drawing")
        self.btn_start_drawing.clicked.connect(self.setup_drawing)
        self.btn_start_drawing.setVisible(False)
        self.layout.addWidget(self.btn_start_drawing)

        self.btn_pause_resume = QPushButton("Pause/Resume Drawing")
        self.btn_pause_resume.clicked.connect(self.pause_or_resume_drawing)
        self.btn_pause_resume.setVisible(False)
        self.layout.addWidget(self.btn_pause_resume)

        self.btn_fill_color = QPushButton("Fill Color")
        self.btn_fill_color.clicked.connect(self.fill_color)
        self.btn_fill_color.setVisible(False)
        self.layout.addWidget(self.btn_fill_color)

        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        self.selected_image = None
        self.selected_image_path = None
        self.resized_image = None
        self.pure_sketch = None

        self.start_x = 0
        self.start_y = 0

        self.display_img = None
        self.all_drawing_instructions = []

        self.line_thickness = 1
        self.draw_batch_size = 50

        self.timer = QTimer()
        self.timer.timeout.connect(self.draw_next_pixels)
        self.current_layer_index = 0
        self.current_pixel_index = 0

        self.is_paused = False

    def choose_image_source(self):
        source = self.source_combo.currentText()
        if "Local" in source:
            self.choose_local_image()
        elif "URL" in source:
            self.choose_url_image()
        elif "Camera" in source:
            self.choose_camera_image()

    def choose_local_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image files (*.png *.jpg *.jpeg)')
        if fname:
            img = cv2.imread(fname)
            if img is not None:
                self.selected_image = img
                self.label_info.setText("Selected image from local: " + fname)
                self.btn_resize.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", "Unable to read the image from local.")

    def choose_url_image(self):
        url, ok = QInputDialog.getText(self, "Enter URL", "URL:")
        if ok and url:
            try:
                img = self.load_image_from_url(url)
                if img is not None:
                    self.selected_image = img
                    self.label_info.setText("Selected image from URL: " + url)
                    self.btn_resize.setEnabled(True)
                else:
                    QMessageBox.warning(self, "Error", "Unable to download the image from URL.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to download the image from URL.\n{e}")

    def choose_camera_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open the camera.")
            return
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.selected_image = frame
            self.label_info.setText("Captured image from camera")
            self.btn_resize.setEnabled(True)
        else:
            QMessageBox.warning(self, "Error", "Failed to capture image from camera.")

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
            QMessageBox.warning(self, "Warning", "You have not selected an image!")
            return

        w, h = self.selected_image.shape[1], self.selected_image.shape[0]
        new_width, ok = QInputDialog.getInt(self, "Enter New Width", "Width:", value=w, min=10, max=10000)
        if ok:
            ratio = new_width / w
            new_height = int(h * ratio)
            self.resized_image = cv2.resize(self.selected_image, (new_width, new_height))
            input_image_path = os.path.join(self.data_folder, "input_img.jpg")
            cv2.imwrite(input_image_path, self.resized_image)
            self.selected_image_path = input_image_path
            self.btn_upload.setEnabled(True)
            self.btn_to_sketch.setEnabled(True)
            self.label_info.setText(f"Resized image to {new_width}x{new_height}")

    def upload_to_kaggle(self):
        if self.resized_image is None or self.selected_image_path is None:
            QMessageBox.warning(self, "Warning", "No image available for upload!")
            return

        print("Uploading image to Kaggle dataset...")
        self.data_handler.upload_image("thuanngoquoc/inputdata", self.selected_image_path)
        print("Upload successful.")

    def convert_to_sketch(self):
        if self.resized_image is None:
            QMessageBox.warning(self, "Warning", "No resized image available for sketch conversion!")
            return

        pure_sketch = self.image_processor.to_sketch(self.resized_image)
        sketch_path = os.path.join(self.sketch_data_folder, "final_sketch.jpg")
        cv2.imwrite(sketch_path, pure_sketch)
        self.pure_sketch = pure_sketch.copy()

        bg_h, bg_w = self.background_image.shape[:2]
        h, w = self.pure_sketch.shape[:2]
        self.start_x = (bg_w - w)//2
        self.start_y = (bg_h - h)//2

        composed = self.background_image.copy()
        composed[self.start_y:self.start_y+h, self.start_x:self.start_x+w] = self.pure_sketch

        self.display_img = composed.copy()

        self.show_image(self.display_img, self.image_label)
        self.image_label.setVisible(True)

        self.btn_start_drawing.setVisible(True)
        self.btn_start_drawing.setEnabled(True)
        self.btn_pause_resume.setVisible(True)
        self.btn_pause_resume.setEnabled(False)
        self.btn_fill_color.setVisible(True)
        self.btn_fill_color.setEnabled(False)

    def show_image(self, img, label_widget):
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, 3*w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        label_widget.setPixmap(pixmap)
        label_widget.setAlignment(Qt.AlignCenter)

    def setup_drawing(self):
        thickness, ok = QInputDialog.getInt(self, "Drawing Thickness", "Enter thickness of the line (>=1):", value=1, min=1, max=20)
        if ok:
            self.line_thickness = thickness

        speed, ok = QInputDialog.getInt(self, "Drawing Speed", "Enter number of pixels to draw each time (lower is slower):", value=50, min=1, max=1000)
        if ok:
            self.draw_batch_size = speed

        self.start_drawing()

    def start_drawing(self):
        result_file = "result.npz"
        if not os.path.exists(result_file):
            QMessageBox.warning(self, "Error", "Segment result.npz file not found!")
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

        h, w = self.pure_sketch.shape[:2]
        self.display_img[self.start_y:self.start_y+h, self.start_x:self.start_x+w] = (255,255,255)
        self.show_image(self.display_img, self.image_label)

        pure_sketch_gray = cv2.cvtColor(self.pure_sketch, cv2.COLOR_BGR2GRAY)
        self.all_drawing_instructions = []
        for label, parts in final_objects:
            for part_name, mask in parts:
                layers = self.extract_layers_from_mask(mask, pure_sketch_gray)
                for layer_pixels in layers:
                    offset_layer = [(x+self.start_x, y+self.start_y) for (x,y) in layer_pixels]
                    self.all_drawing_instructions.append((label, part_name, offset_layer))

        remaining_sketch_pixels = self.extract_remaining_sketch_pixels(pure_sketch_gray)
        if remaining_sketch_pixels:
            self.all_drawing_instructions.append(("remaining_sketch", "remaining", remaining_sketch_pixels))

        self.current_layer_index = 0
        self.current_pixel_index = 0
        self.timer.start(20)
        self.btn_pause_resume.setEnabled(True)

    def pause_or_resume_drawing(self):
        if self.is_paused:
            self.timer.start(20)
            self.is_paused = False
            self.btn_pause_resume.setText("Pause Drawing")
        else:
            self.timer.stop()
            self.is_paused = True
            self.btn_pause_resume.setText("Resume Drawing")

    def draw_next_pixels(self):
        if self.current_layer_index >= len(self.all_drawing_instructions):
            self.timer.stop()
            self.btn_pause_resume.setEnabled(False)
            self.btn_fill_color.setEnabled(True)
            return

        pixels = self.all_drawing_instructions[self.current_layer_index][2]
        end_index = min(self.current_pixel_index + self.draw_batch_size, len(pixels))
        for i in range(self.current_pixel_index, end_index):
            x, y = pixels[i]
            self.draw_pixel_with_thickness(x, y, (0,0,0), self.line_thickness)

        self.current_pixel_index = end_index
        self.show_image(self.display_img, self.image_label)

        if self.current_pixel_index >= len(pixels):
            self.current_layer_index += 1
            self.current_pixel_index = 0
            if self.current_layer_index >= len(self.all_drawing_instructions):
                self.timer.stop()
                self.btn_pause_resume.setEnabled(False)
                self.btn_fill_color.setEnabled(True)

    def draw_pixel_with_thickness(self, x, y, color, thickness=1):
        half = thickness//2
        h, w, c = self.display_img.shape
        x_start = max(x-half, 0)
        y_start = max(y-half, 0)
        x_end = min(x-half+thickness, w)
        y_end = min(y-half+thickness, h)
        self.display_img[y_start:y_end, x_start:x_end] = color

    def extract_layers_from_mask(self, mask, sketch_gray):
        layers = []
        current_mask = mask.copy()
        threshold_black = 50
        kernel = np.ones((3,3), np.uint8)

        while True:
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                break

            contour_pixels = []
            for cnt in contours:
                for pt in cnt:
                    x, y = pt[0]
                    if sketch_gray[y, x] < threshold_black:
                        contour_pixels.append((x,y))

            if len(contour_pixels) > 0:
                contour_pixels.sort(key=lambda p: p[1]*10000+p[0])
                layers.append(contour_pixels)

            eroded = cv2.erode(current_mask, kernel, iterations=1)
            if np.all(eroded == 0):
                break
            current_mask = eroded

        return layers

    def extract_remaining_sketch_pixels(self, sketch_gray):
        remaining_pixels = []
        for y in range(sketch_gray.shape[0]):
            for x in range(sketch_gray.shape[1]):
                if sketch_gray[y, x] < 50:
                    remaining_pixels.append((x + self.start_x, y + self.start_y))
        return remaining_pixels

    def fill_color(self):
        if self.selected_image is None or self.pure_sketch is None or self.selected_image_path is None:
            QMessageBox.warning(self, "Error", "No image or sketch available for filling color!")
            return
    
        input_img = cv2.imread(self.selected_image_path)
        if input_img is None:
            QMessageBox.warning(self, "Error", "Failed to load input image for coloring!")
            return
    
        input_img = cv2.resize(input_img, (self.pure_sketch.shape[1], self.pure_sketch.shape[0]))
    
        result_file = "result.npz"
        if not os.path.exists(result_file):
            QMessageBox.warning(self, "Error", "Segment result.npz file not found!")
            return
    
        data = np.load(result_file, allow_pickle=True)
        objects_data = data["objects"].item()
        objects = objects_data["objects"]
    
        self.coloring_instructions = []
    
        # Prepare coloring instructions for each segment
        for obj in objects:
            parts = obj["parts"]
            if isinstance(parts, np.ndarray):
                parts = parts.item()  # Handle NumPy structured data
            for part_name, mask in parts.items():  # Assume `parts` is now a dictionary
                mask_pixels = list(zip(*np.where(mask > 0)))
                if mask_pixels:
                    self.coloring_instructions.append(("segment", mask_pixels))
    
        # Add remaining sketch areas
        remaining_pixels = self.extract_remaining_sketch_pixels(cv2.cvtColor(self.pure_sketch, cv2.COLOR_BGR2GRAY))
        if remaining_pixels:
            self.coloring_instructions.append(("remaining", remaining_pixels))
    
        self.input_img = input_img
        self.current_coloring_index = 0
        self.current_pixel_index = 0
        self.timer.timeout.connect(self.color_next_pixels)
        self.timer.start(20)
    
    def color_next_pixels(self):
        if self.current_coloring_index >= len(self.coloring_instructions):
            self.timer.stop()
            QMessageBox.information(self, "Fill Color", "Coloring complete!")
            return
    
        segment_type, pixels = self.coloring_instructions[self.current_coloring_index]
        batch_size = 10000  # Number of pixels to color in each batch
        end_index = min(self.current_pixel_index + batch_size, len(pixels))
    
        for i in range(self.current_pixel_index, end_index):
            y, x = pixels[i]
            if segment_type == "segment":
                self.display_img[y + self.start_y, x + self.start_x] = self.input_img[y, x]
            elif segment_type == "remaining":
                self.display_img[y, x] = self.input_img[y - self.start_y, x - self.start_x]
    
        self.current_pixel_index = end_index
        self.show_image(self.display_img, self.image_label)
    
        if self.current_pixel_index >= len(pixels):
            self.current_coloring_index += 1
            self.current_pixel_index = 0

               
