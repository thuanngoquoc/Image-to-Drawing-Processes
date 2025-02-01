import os
import time
from kaggle.api.kaggle_api_extended import KaggleApi
from tkinter import filedialog, Label
import tkinter as tk
from PIL import Image, ImageTk
import cv2

# Khởi tạo API của Kaggle
api = KaggleApi()
api.authenticate()

# Đường dẫn cho dataset và tải lên/output tại máy cục bộ
DATASET_NAME = "thuanngoquoc/oneformerdataset"
UPLOAD_FOLDER = "E:/My_projects/AI_Projects/imgseg/Image-to-Drawing-Processes/upload_images"
OUTPUT_PATH = "E:/My_projects/AI_Projects/imgseg/Image-to-Drawing-Processes/local_output_path/segmented_output.png"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm để di chuyển ảnh đã chọn hoặc chụp vào thư mục upload
def upload_image_to_kaggle(image_path):
    img_name = os.path.basename(image_path)
    target_path = os.path.join(UPLOAD_FOLDER, img_name)
    
    # Di chuyển ảnh vào thư mục upload
    if image_path != target_path:
        os.rename(image_path, target_path)

    # Kiểm tra nếu `dataset-metadata.json` có trong thư mục upload
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'dataset-metadata.json')):
        print("Lỗi: dataset-metadata.json không có trong UPLOAD_FOLDER.")
        return

    # Tải lên phiên bản dataset
    try:
        api.dataset_create_version(UPLOAD_FOLDER, version_notes="New input image uploaded", delete_old_versions=True)
        print(f"Ảnh {img_name} đã tải lên thành công.")
    except Exception as e:
        print(f"Lỗi khi upload dataset: {e}")

# Code giao diện - Chọn hoặc Chụp Ảnh, Xử lý và Hiển thị Kết quả
def select_image():
    global img_path
    img_path = filedialog.askopenfilename()
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        input_image_label.config(image=img_tk)
        input_image_label.image = img_tk

def capture_image():
    global img_path
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        img_path = os.path.join(UPLOAD_FOLDER, "captured_image.jpg")
        cv2.imwrite(img_path, frame)
        img = Image.open(img_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        input_image_label.config(image=img_tk)
        input_image_label.image = img_tk
    cap.release()

def process_image():
    upload_image_to_kaggle(img_path)

root = tk.Tk()
root.title("Giao diện phân đoạn tự động")

# Cài đặt các frame
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

input_image_label = Label(input_frame)
input_image_label.pack()

# Nút để chọn hoặc chụp ảnh
select_btn = tk.Button(input_frame, text="Chọn Ảnh", command=select_image)
select_btn.pack(side="left", padx=5)
capture_btn = tk.Button(input_frame, text="Chụp Ảnh", command=capture_image)
capture_btn.pack(side="left", padx=5)

process_btn = tk.Button(root, text="Xử lý Ảnh", command=process_image)
process_btn.pack(pady=10)

root.mainloop()
