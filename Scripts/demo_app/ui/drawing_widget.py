from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
import os

class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.timer = QTimer()
        self.instructions = []
        self.current_obj_index = 0
        self.current_pixel_index = 0
        self.current_pixels = []
        self.base_img = None
        self.display_img = None

    def set_data(self, objects_parts_data, sketch_img):
        # Hiện chỉ cần hiển thị sketch_img
        self.display_img = sketch_img.copy()
        self.update_display()

    def update_display(self):
        if self.display_img is not None:
            h, w, c = self.display_img.shape
            qimg = QImage(self.display_img.data, w, h, 3*w, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(pixmap)

    def set_drawing_instructions(self, instructions, base_img):
        self.instructions = instructions
        self.base_img = base_img.copy()
        self.display_img = base_img.copy()
        self.current_obj_index = 0
        self.current_pixel_index = 0
        if self.instructions:
            self.current_pixels = self.instructions[0][2]
        else:
            self.current_pixels = []
        self.update_display()

    def start_gradual_drawing(self):
        self.timer.timeout.connect(self.draw_next_pixels)
        self.timer.start(20)

    def draw_next_pixels(self):
        if self.current_obj_index >= len(self.instructions):
            self.timer.stop()
            return
        pixels = self.current_pixels
        batch_size = 50
        end_index = min(self.current_pixel_index + batch_size, len(pixels))
        for i in range(self.current_pixel_index, end_index):
            x, y = pixels[i]
            self.display_img[y, x] = (0,0,0)
        self.current_pixel_index = end_index
        self.update_display()

        if self.current_pixel_index >= len(pixels):
            self.current_obj_index += 1
            if self.current_obj_index < len(self.instructions):
                self.current_pixels = self.instructions[self.current_obj_index][2]
                self.current_pixel_index = 0
            else:
                self.timer.stop()

    def clear_sketch_area(self, start_x, start_y, w, h):
        if self.display_img is not None:
            self.display_img[start_y:start_y+h, start_x:start_x+w] = (255,255,255)
            self.update_display()
