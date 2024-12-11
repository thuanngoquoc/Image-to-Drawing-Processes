from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
import time
from playsound import playsound
import os

class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.label = QLabel(self)
        self.start_button = QPushButton("Bắt đầu vẽ", self)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

        self.start_button.clicked.connect(self.start_drawing)
        self.timer = QTimer()
        self.timer.timeout.connect(self.draw_next_line_segment)

        # sẽ được set từ bên ngoài
        self.objects_parts_data = []
        self.original_img = None
        self.display_img = None
        self.current_object_index = 0
        self.current_part_index = 0
        self.current_contour_points = []
        self.current_point_index = 0

    def set_data(self, objects_parts_data, sketch_img):
        self.objects_parts_data = objects_parts_data
        self.original_img = sketch_img.copy()
        self.display_img = sketch_img.copy()
        self.current_object_index = 0
        self.current_part_index = 0
        self.update_display()

    def update_display(self):
        h, w, c = self.display_img.shape
        qimg = QImage(self.display_img.data, w, h, 3*w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def start_drawing(self):
        if not self.objects_parts_data:
            return
        # kiểm tra object hiện tại có phải đặc biệt không
        class_name = self.objects_parts_data[self.current_object_index][0]
        if class_name in ["dog", "cat", "teddy bear"]:
            self.draw_guidelines()
            self.update_display()
            time.sleep(1)
        self.draw_next_part()

    def draw_guidelines(self):
        h, w, _ = self.display_img.shape
        cv2.line(self.display_img, (w//2,0), (w//2,h), (0,0,0), 1)
        cv2.line(self.display_img, (0,h//2), (w,h//2), (0,0,0), 1)

    def draw_next_part(self):
        if self.current_object_index >= len(self.objects_parts_data):
            return
        class_name, parts = self.objects_parts_data[self.current_object_index]
        if self.current_part_index >= len(parts):
            # chuyển sang object tiếp theo
            self.current_object_index += 1
            self.current_part_index = 0
            if self.current_object_index < len(self.objects_parts_data):
                new_class_name, _ = self.objects_parts_data[self.current_object_index]
                if new_class_name in ["dog", "cat", "teddy bear"]:
                    self.draw_guidelines()
                    self.update_display()
                    time.sleep(1)
                self.draw_next_part()
            return

        part_name, mask = parts[self.current_part_index]

        # phát âm thanh
        sound_file = os.path.join("sound", f"vẽ_{part_name}.mp3")
        if os.path.exists(sound_file):
            playsound(sound_file, False)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            self.current_part_index += 1
            self.draw_next_part()
            return

        contour = max(contours, key=cv2.contourArea)
        contour = contour.squeeze(axis=1)
        if len(contour.shape) < 2: 
            # có thể contour chỉ có 1 điểm
            self.current_part_index += 1
            self.draw_next_part()
            return

        self.current_contour_points = contour
        self.current_point_index = 0
        self.timer.start(20)  # vẽ dần dần

    def draw_next_line_segment(self):
        pts = self.current_contour_points
        if self.current_point_index < len(pts)-1:
            pt1 = tuple(pts[self.current_point_index])
            pt2 = tuple(pts[self.current_point_index+1])
            cv2.line(self.display_img, pt1, pt2, (0,0,0), 2)
            self.current_point_index += 1
            self.update_display()
        else:
            self.timer.stop()
            self.current_part_index += 1
            QTimer.singleShot(500, self.draw_next_part)
