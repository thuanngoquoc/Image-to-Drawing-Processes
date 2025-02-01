import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import cv2
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import time

# Khởi tạo Kaggle API
api = KaggleApi()
api.authenticate()

# Đường dẫn dataset và notebook trên Kaggle
DATASET_NAME = "thuanngoquoc/oneformerdataset"
NOTEBOOK_SLUG = "thuanngoquoc/oneformertest3"

# Đường dẫn upload ảnh và đầu ra
UPLOAD_FOLDER = os.path.abspath("E:/My_projects/AI_Projects/imgseg/Image-to-Drawing-Processes/upload_images")
OUTPUT_PATH = os.path.abspath("E:/My_projects/AI_Projects/imgseg/Image-to-Drawing-Processes/local_output_path/segmented_output.png")

# Đảm bảo thư mục upload tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm upload ảnh lên Kaggle
def upload_image_to_kaggle(image_path):
    if not image_path:
        print("Error: No image selected.")
        return
    
    img_name = os.path.basename(image_path)
    target_path = os.path.join(UPLOAD_FOLDER, img_name)
    
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'dataset-metadata.json')):
        print("Error: dataset-metadata.json not found in UPLOAD_FOLDER.")
        return
    
    # Sao chép ảnh vào thư mục upload
    if image_path != target_path:
        os.replace(image_path, target_path)
    
    api.dataset_create_version(UPLOAD_FOLDER, version_notes="New input image uploaded")

# Kích hoạt notebook tự động
#def start_kaggle_notebook():
#    api.kernels_start(NOTEBOOK_SLUG)
#    print(f"Notebook {NOTEBOOK_SLUG} started.")

# Theo dõi trạng thái notebook
def check_kernel_status():
    while True:
        status = api.kernels_status(NOTEBOOK_SLUG)
        if status['status'] == 'complete':
            print("Processing complete.")
            break
        elif status['status'] == 'error':
            print("Kernel failed to execute.")
            break
        else:
            print("Kernel is still running...")
            time.sleep(30)

# Tải file kết quả từ Kaggle về
def download_output():
    try:
        api.dataset_download_file(DATASET_NAME, "segmented_output.png", path=os.path.dirname(OUTPUT_PATH))
        print("Output downloaded successfully.")
    except:
        print("Output not ready yet. Retrying...")
        time.sleep(30)
        download_output()

# Hàm chọn ảnh
def select_image():
    global img_path
    img_path = filedialog.askopenfilename()
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        input_image_label.config(image=img_tk)
        input_image_label.image = img_tk

# Hàm chụp ảnh từ camera
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

# Xử lý toàn bộ quy trình
def process_image():
    if not img_path:
        print("No image selected for processing.")
        return
    
    upload_image_to_kaggle(img_path)
    #start_kaggle_notebook()
    check_kernel_status()
    download_output()
    display_output()

# Hiển thị ảnh output
def display_output():
    if os.path.exists(OUTPUT_PATH):
        img = Image.open(OUTPUT_PATH)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        output_image_label.config(image=img_tk)
        output_image_label.image = img_tk

# Giao diện Tkinter
root = tk.Tk()
root.title("Automatic Segmentation Interface")

# Khung input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Label hiển thị ảnh input
input_image_label = Label(input_frame)
input_image_label.pack()

# Nút chọn ảnh
select_btn = tk.Button(input_frame, text="Select Image", command=select_image)
select_btn.pack(side="left", padx=5)

# Nút chụp ảnh
capture_btn = tk.Button(input_frame, text="Capture Image", command=capture_image)
capture_btn.pack(side="left", padx=5)

# Nút xử lý ảnh
process_btn = tk.Button(root, text="Process Image", command=process_image)
process_btn.pack(pady=10)

# Khung output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)

# Label hiển thị ảnh output
output_image_label = Label(output_frame)
output_image_label.pack()

root.mainloop()
