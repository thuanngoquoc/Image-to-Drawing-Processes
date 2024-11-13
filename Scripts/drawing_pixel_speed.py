import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk

# 1. Chuyển đổi ảnh màu sang ảnh grayscale
img_path = 'E:\My_projects\AI_Projects\imgseg\Image-to-Drawing-Processes\image\img7.jpg'
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
invert = cv2.bitwise_not(grey_img)
blur = cv2.GaussianBlur(invert, (21, 21), 0)
invertedblur = cv2.bitwise_not(blur)
sketch = cv2.divide(grey_img, invertedblur, scale=256.0)
#cv2.imshow("dd", sketch)

# 2. Điều chỉnh kích thước canvas
PAPER_SIZES = {
    'A4': (210 * 96 / 25.4, 297 * 96 / 25.4),
    'A3': (297 * 96 / 25.4, 420 * 96 / 25.4),
}
ORIENTATIONS = ['Portrait', 'Landscape']

def create_canvas(size, orientation):
    width, height = size
    if orientation == 'Landscape':
        width, height = height, width
    canvas.config(width=int(width), height=int(height))
    canvas.pack()

def apply_settings():
    size = PAPER_SIZES[size_var.get()]
    orientation = orientation_var.get()
    create_canvas(size, orientation)

root = tk.Tk()
root.title("Canvas Size Selector")

# Tạo các widget để chọn kích thước và hướng
size_var = tk.StringVar(value='A4')
orientation_var = tk.StringVar(value='Portrait')

tk.Label(root, text="Select Paper Size:").pack()
size_menu = ttk.Combobox(root, textvariable=size_var, values=list(PAPER_SIZES.keys()))
size_menu.pack()

tk.Label(root, text="Select Orientation:").pack()
orientation_menu = ttk.Combobox(root, textvariable=orientation_var, values=ORIENTATIONS)
orientation_menu.pack()

apply_button = tk.Button(root, text="Apply", command=apply_settings)
apply_button.pack()

# Tạo canvas ban đầu
canvas = tk.Canvas(root)
canvas.pack()

# 3. Tạo ngưỡng đen xám
black_threshold = 0
gray_upper_bound = 225
gray_mask = (sketch >= black_threshold) & (sketch <= gray_upper_bound)
gray_coords = np.column_stack(np.where(gray_mask))

# 4. Vẽ điểm hình oval theo tọa độ pixel
stroke_color = tk.StringVar(value="black")
point_size = 1
delay = 1  # Reduced delay for faster drawing
points_per_iteration = 10  # Number of points to draw per iteration
white_pixel_threshold = 240

# Load pen image
pen_img = Image.open("pencil.png")  # Path to your pen image
pen_img = pen_img.resize((40, 40), Image.LANCZOS)
pen_tk = ImageTk.PhotoImage(pen_img)

# Determine the offset to position the tip of the pen at the coordinates
pen_tip_offset_x = 10  # Adjust based on the actual tip position in your pen image
pen_tip_offset_y = 10  # Adjust based on the actual tip position in your pen image

pen_id = None

def draw_points(coords, index=0):
    global pen_id
    if index < len(coords):
        for _ in range(points_per_iteration):
            if index >= len(coords):
                break
            point = coords[index]
            if sketch[point[0], point[1]] <= white_pixel_threshold:
                x, y = point[1], point[0]

                if pen_id is not None:
                    canvas.delete(pen_id)

                pen_id = canvas.create_image(
                    x - pen_tip_offset_x, y - pen_tip_offset_y,
                    image=pen_tk, anchor=tk.CENTER
                )
                canvas.create_oval(
                    x - point_size // 2, y - point_size // 2,
                    x + point_size // 2, y + point_size // 2,
                    fill=stroke_color.get(), outline=stroke_color.get()
                )
            index += 1
        root.after(delay, draw_points, coords, index)
    else:
        if pen_id is not None:
            canvas.delete(pen_id)

if len(gray_coords) > 0:
    draw_points(gray_coords)

# Start the Tkinter main loop
root.mainloop()
