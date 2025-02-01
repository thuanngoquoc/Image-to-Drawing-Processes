import os
import cv2
import numpy as np
import requests

from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

###############################################
# STUB CLASSES (giả lập) - Thay thế bằng code thực của bạn
###############################################

class KaggleDataHandler:
    def upload_image(self, dataset_slug, image_path):
        """
        Giả lập upload hình lên Kaggle dataset. 
        Trong thực tế sẽ sử dụng Kaggle API.
        """
        print(f"[KaggleDataHandler] Uploading {image_path} -> {dataset_slug}")
        # ... Thực hiện upload ...

class PriorityHandler:
    def __init__(self, priority_map):
        self.priority_map = priority_map if priority_map else {}
    
    def sort_parts(self, class_name, parts):
        """
        Trả về danh sách (part_name, mask) đã được sắp xếp 
        theo thứ tự ưu tiên (nếu có) do priority_map quy định.
        """
        # Nếu class_name không có trong priority_map, ta trả về tất cả parts như cũ.
        if class_name not in self.priority_map:
            return list(parts.items())
        
        priority_list = self.priority_map[class_name]  # ví dụ ["head", "ear", ...]
        sorted_parts = []

        # Bước 1: Gom part có trong priority_list vào trước, đúng thứ tự
        used_part_names = set()
        for p_name in priority_list:
            if p_name in parts:
                sorted_parts.append((p_name, parts[p_name]))
                used_part_names.add(p_name)

        # Bước 2: Phần còn lại (không có trong priority_list) cho vào sau
        for p_name, mask in parts.items():
            if p_name not in used_part_names:
                sorted_parts.append((p_name, mask))

        return sorted_parts

class ImageProcessor:
    def to_sketch(self, image_bgr):
        """
        Chuyển ảnh BGR thành ảnh sketch (giả lập). 
        Thực tế bạn có thể xài bộ lọc edge detection.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Thực hiện đơn giản: invert + threshold
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        inv_blur = 255 - blur
        sketch = cv2.divide(gray, inv_blur, scale=256)
        # Kết quả là ảnh xám, để trả về BGR ta stack
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        return sketch_bgr

class KaggleSegmentProcessor:
    def __init__(self, data_handler, dataset_slug):
        """
        Giả lập class xử lý phân mảng (segmentation).
        Ở đây để trống, tuỳ dự án của bạn.
        """
        self.data_handler = data_handler
        self.dataset_slug = dataset_slug

    # Bạn có thể viết các hàm segment(...) nếu cần.
    # Code gốc chưa thấy gọi trực tiếp nên tạm để trống.

###############################################
# PHẦN CHÍNH: CLASS MainWindow
###############################################

class MainWindow(QMainWindow):
    def __init__(self, priority_map=None, dataset_slug="thuanngoquoc/inputdata"):
        super().__init__()
        self.setWindowTitle("Image Selection and Drawing App")

        if priority_map is None:
            priority_map = {
                # Ví dụ (tuỳ ý bạn):
                # "dog": ["head", "ear", "eye", "mouth", "body", "front_leg", "hind_leg"]
            }

        self.data_handler = KaggleDataHandler()
        self.priority_handler = PriorityHandler(priority_map)
        self.image_processor = ImageProcessor()
        self.segment_processor = KaggleSegmentProcessor(self.data_handler, dataset_slug)

        self.data_folder = "data_folder"
        os.makedirs(self.data_folder, exist_ok=True)

        self.sketch_data_folder = "sketch_data"
        os.makedirs(self.sketch_data_folder, exist_ok=True)

        # Load ảnh nền (nếu có)
        self.background_image = cv2.imread("background.jpg")
        if self.background_image is None:
            self.background_image = np.ones((1920, 1080, 3), dtype=np.uint8)*255
        else:
            self.background_image = cv2.resize(self.background_image, (1080, 1920))

        # Tạo layout giao diện
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

        # Biến để lưu trữ ảnh được chọn, v.v.
        self.selected_image = None
        self.selected_image_path = None
        self.resized_image = None
        self.pure_sketch = None

        # Toạ độ đặt ảnh sketch lên background
        self.start_x = 0
        self.start_y = 0

        self.display_img = None

        # Danh sách hướng dẫn vẽ
        # Mỗi phần tử là (object_label, part_name, [contour1, contour2, ...])
        self.all_drawing_instructions = []

        # Các biến cho quá trình vẽ:
        self.line_thickness = 1
        self.timer = QTimer()
        self.timer.timeout.connect(self.draw_next_line)

        # Các chỉ số dùng để duyệt contour
        self.current_obj_index = 0
        self.current_contour_index = 0
        self.current_point_index = 0

        # Trạng thái tạm dừng
        self.is_paused = False

        # Số đoạn line muốn vẽ mỗi lần
        self.draw_batch_size = 1

        # Dùng cho bước tô màu
        self.coloring_instructions = []
        self.current_coloring_index = 0
        self.current_pixel_index = 0
        self.input_img = None

    ######################################
    # Các hàm chọn ảnh từ Local/URL/Camera
    ######################################
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

    ######################################
    # Resize và upload ảnh
    ######################################
    def resize_image_dialog(self):
        if self.selected_image is None:
            QMessageBox.warning(self, "Warning", "You have not selected an image!")
            return

        w, h = self.selected_image.shape[1], self.selected_image.shape[0]
        new_width, ok = QInputDialog.getInt(
            self, "Enter New Width", "Width:", value=w, min=10, max=10000
        )
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

    ######################################
    # Chuyển sang sketch
    ######################################
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

    ######################################
    # Thiết lập và bắt đầu quá trình vẽ nét
    ######################################
    def setup_drawing(self):
        thickness, ok = QInputDialog.getInt(
            self, "Drawing Thickness", 
            "Enter thickness of the line (>=1):", 
            value=1, min=1, max=20
        )
        if ok:
            self.line_thickness = thickness

        speed, ok = QInputDialog.getInt(
            self, "Drawing Speed", 
            "Enter number of line segments to draw each time (lower is slower):", 
            value=1, min=1, max=50
        )
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

        # Đếm số lần xuất hiện class để đặt label (nếu cần)
        class_counts = {}
        for obj in objects:
            c_name = obj["class_name"]
            class_counts[c_name] = class_counts.get(c_name, 0) + 1
            obj["label"] = f"{c_name}{class_counts[c_name]}"

        # Tạo list (label, sorted_parts)
        final_objects = []
        for obj in objects:
            c_name = obj["class_name"]
            parts = obj["parts"].item() if isinstance(obj["parts"], np.ndarray) else obj["parts"]
            sorted_parts = self.priority_handler.sort_parts(c_name, parts)
            final_objects.append((obj["label"], sorted_parts))

        # Xoá vùng sketch trên display_img (cho trắng hết) trước khi vẽ
        h, w = self.pure_sketch.shape[:2]
        self.display_img[self.start_y:self.start_y+h, self.start_x:self.start_x+w] = (255, 255, 255)
        self.show_image(self.display_img, self.image_label)

        # Chuẩn bị danh sách hướng dẫn vẽ
        self.all_drawing_instructions = []
        pure_sketch_gray = cv2.cvtColor(self.pure_sketch, cv2.COLOR_BGR2GRAY)

        for label, parts in final_objects:
            for part_name, mask in parts:
                if mask is None:
                    continue
                # Lấy contour
                contour_list = self.extract_contours_in_order(mask, pure_sketch_gray)
                self.all_drawing_instructions.append((label, part_name, contour_list))

        # Thêm vùng còn sót lại (nếu cần)
        remaining_mask = self.get_remaining_mask(pure_sketch_gray, objects)
        if remaining_mask is not None:
            remain_contours = self.extract_contours_in_order(remaining_mask, pure_sketch_gray)
            if remain_contours:
                self.all_drawing_instructions.append(("remaining_sketch", "remaining", remain_contours))

        # Khởi tạo chỉ số
        self.current_obj_index = 0
        self.current_contour_index = 0
        self.current_point_index = 0

        # Bắt đầu timer
        self.timer.start(30)
        self.btn_pause_resume.setEnabled(True)

    def pause_or_resume_drawing(self):
        if self.is_paused:
            self.timer.start(30)
            self.is_paused = False
            self.btn_pause_resume.setText("Pause Drawing")
        else:
            self.timer.stop()
            self.is_paused = True
            self.btn_pause_resume.setText("Resume Drawing")

    ######################################
    # Hàm vẽ nét “line-by-line”
    ######################################
    def draw_next_line(self):
        # Kiểm tra đã vẽ hết mọi thứ chưa
        if self.current_obj_index >= len(self.all_drawing_instructions):
            self.timer.stop()
            self.btn_pause_resume.setEnabled(False)
            self.btn_fill_color.setEnabled(True)
            return

        # Lấy ra danh sách contour của object-part hiện tại
        _, _, contour_list = self.all_drawing_instructions[self.current_obj_index]

        if self.current_contour_index >= len(contour_list):
            # chuyển sang object-part tiếp theo
            self.current_obj_index += 1
            self.current_contour_index = 0
            self.current_point_index = 0
            return

        # Lấy contour hiện tại
        current_contour = contour_list[self.current_contour_index]
        if len(current_contour) < 2:
            # Nếu contour quá ngắn, bỏ qua
            self.current_contour_index += 1
            self.current_point_index = 0
            return

        # Vẽ 1 batch (self.draw_batch_size) đoạn line
        for _ in range(self.draw_batch_size):
            if self.current_point_index >= len(current_contour) - 1:
                # Hết contour -> sang contour tiếp
                self.current_contour_index += 1
                self.current_point_index = 0
                self.show_image(self.display_img, self.image_label)
                return

            (x1, y1) = current_contour[self.current_point_index]
            (x2, y2) = current_contour[self.current_point_index + 1]

            # Dịch toạ độ
            x1_display = x1 + self.start_x
            y1_display = y1 + self.start_y
            x2_display = x2 + self.start_x
            y2_display = y2 + self.start_y

            # Vẽ line
            cv2.line(
                self.display_img,
                (x1_display, y1_display),
                (x2_display, y2_display),
                (0, 0, 0),
                thickness=self.line_thickness
            )

            self.current_point_index += 1

        # Cập nhật hiển thị
        self.show_image(self.display_img, self.image_label)

    ######################################
    # Hàm tìm contour (chu vi) của mask
    ######################################
    def extract_contours_in_order(self, mask, sketch_gray):
        """
        Trả về list các contour, 
        mỗi contour là list các điểm (x, y) liên tiếp nhau.
        """
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contour_list = []
        for cnt in contours:
            pts = [(p[0][0], p[0][1]) for p in cnt]  # (x, y)
            contour_list.append(pts)
        return contour_list

    def get_remaining_mask(self, sketch_gray, objects):
        """
        Sinh mask cho các pixel đen (<50) không thuộc bất kỳ phần (part) nào.
        """
        base_mask = (sketch_gray < 50).astype(np.uint8)

        for obj in objects:
            parts = obj["parts"].item() if isinstance(obj["parts"], np.ndarray) else obj["parts"]
            for _, pmask in parts.items():
                if pmask is not None:
                    base_mask[pmask > 0] = 0

        if np.count_nonzero(base_mask) == 0:
            return None
        return base_mask

    ######################################
    # Hàm tô màu
    ######################################
    def fill_color(self):
        if self.selected_image is None or self.pure_sketch is None or self.selected_image_path is None:
            QMessageBox.warning(self, "Error", "No image or sketch available for filling color!")
            return

        input_img = cv2.imread(self.selected_image_path)
        if input_img is None:
            QMessageBox.warning(self, "Error", "Failed to load input image for coloring!")
            return

        # Resize ảnh input cho khớp kích thước sketch
        input_img = cv2.resize(input_img, (self.pure_sketch.shape[1], self.pure_sketch.shape[0]))

        result_file = "result.npz"
        if not os.path.exists(result_file):
            QMessageBox.warning(self, "Error", "Segment result.npz file not found!")
            return

        data = np.load(result_file, allow_pickle=True)
        objects_data = data["objects"].item()
        objects = objects_data["objects"]

        self.coloring_instructions = []

        # Tạo danh sách pixel cho phần tô màu
        for obj in objects:
            parts = obj["parts"].item() if isinstance(obj["parts"], np.ndarray) else obj["parts"]
            for _, mask in parts.items():
                if mask is not None:
                    mask_pixels = list(zip(*np.where(mask > 0)))
                    if mask_pixels:
                        self.coloring_instructions.append(("segment", mask_pixels))

        # Thêm vùng còn lại nếu muốn tô hết
        remaining_pixels = self.extract_remaining_sketch_pixels(
            cv2.cvtColor(self.pure_sketch, cv2.COLOR_BGR2GRAY)
        )
        if remaining_pixels:
            self.coloring_instructions.append(("remaining", remaining_pixels))

        self.input_img = input_img
        self.current_coloring_index = 0
        self.current_pixel_index = 0

        # Ngắt kết nối timer cũ và gắn hàm tô màu
        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.color_next_pixels)
        self.timer.start(20)

    def extract_remaining_sketch_pixels(self, sketch_gray):
        remaining_pixels = []
        for y in range(sketch_gray.shape[0]):
            for x in range(sketch_gray.shape[1]):
                if sketch_gray[y, x] < 50:
                    remaining_pixels.append((y, x))
        return remaining_pixels

    def color_next_pixels(self):
        if self.current_coloring_index >= len(self.coloring_instructions):
            self.timer.stop()
            QMessageBox.information(self, "Fill Color", "Coloring complete!")
            return

        segment_type, pixels = self.coloring_instructions[self.current_coloring_index]
        batch_size = 300  # tốc độ tô
        end_index = min(self.current_pixel_index + batch_size, len(pixels))

        for i in range(self.current_pixel_index, end_index):
            y, x = pixels[i]
            if segment_type == "segment":
                self.display_img[y + self.start_y, x + self.start_x] = self.input_img[y, x]
            elif segment_type == "remaining":
                # Lưu ý offset
                self.display_img[y, x] = self.input_img[y - self.start_y, x - self.start_x]

        self.current_pixel_index = end_index
        self.show_image(self.display_img, self.image_label)

        if self.current_pixel_index >= len(pixels):
            self.current_coloring_index += 1
            self.current_pixel_index = 0
