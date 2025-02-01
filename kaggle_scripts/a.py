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

        # (1) Instantiate components
        self.data_handler = KaggleDataHandler()
        self.priority_handler = PriorityHandler(priority_map)
        self.image_processor = ImageProcessor()
        self.segment_processor = KaggleSegmentProcessor(self.data_handler, dataset_slug)

        # (2) Setup folders
        self.data_folder = "data_folder"
        self.sketch_data_folder = "sketch_data"

        # (3) Build the UI
        central_widget = QWidget()
        self.layout = QVBoxLayout()

        self.label_info = QLabel("Select image")
        self.layout.addWidget(self.label_info)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Local", "URL", "Camera"])
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

        # (4) Sketch and Drawing Buttons
        self.btn_to_sketch = QPushButton("Convert to Sketch")
        self.btn_to_sketch.clicked.connect(self.convert_to_sketch)
        self.btn_to_sketch.setEnabled(False)
        self.layout.addWidget(self.btn_to_sketch)

        self.image_label = QLabel()
        self.image_label.setVisible(False)
        self.layout.addWidget(self.image_label)

        # (5) Final assembly
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def choose_image_source(self):
        source = self.source_combo.currentText()
        if "Local" in source:
            self.choose_local_image()
        elif "URL" in source:
            self.choose_url_image()
        elif "Camera" in source:
            self.choose_camera_image()