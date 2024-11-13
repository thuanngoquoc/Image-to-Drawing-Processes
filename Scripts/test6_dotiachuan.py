import tkinter as tk
from tkinter import colorchooser
import numpy as np
import cv2

# 1. Chuyển đổi ảnh màu sang ảnh grayscale
img = cv2.imread('E:\My_projects\AI_Projects\imgseg\Image-to-Drawing-Processes\image\img7.jpg', cv2.IMREAD_UNCHANGED)
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
invert = cv2.bitwise_not(grey_img)
blur = cv2.GaussianBlur(invert, (21, 21), 0)
invertedblur = cv2.bitwise_not(blur)
sketch = cv2.divide(grey_img, invertedblur, scale=400.0)

# 2. Điều chỉnh kích thước canvas
root = tk.Tk()
canvas = tk.Canvas(root, width=1100, height=1100)
canvas.pack()

# 3. Cài đặt thông số dò tia
stroke_color = tk.StringVar(value="black")
point_size = 3
delay = 1
white_pixel_threshold = 240  # Ngưỡng để xác định pixel trắng
num_rays = 360  # Số lượng tia để quét
ray_origin = (sketch.shape[1] // 2, sketch.shape[0] // 2)  # Tâm ảnh

def ray_casting(sketch, origin, angle):
    x0, y0 = origin
    coords = []
    for i in range(max(sketch.shape)):
        x = int(x0 + i * np.cos(np.radians(angle)))
        y = int(y0 + i * np.sin(np.radians(angle)))
        if 0 <= x < sketch.shape[1] and 0 <= y < sketch.shape[0]:
            if sketch[y, x] <= white_pixel_threshold:
                coords.append((y, x))
        else:
            break
    return coords

# 4. Tạo danh sách các điểm từ các tia
ray_points = []
for angle in np.linspace(0, 360, num_rays, endpoint=False):
    ray_points.extend(ray_casting(sketch, ray_origin, angle))

def draw_points(coords, index=0):
    if index < len(coords):
        point = coords[index]
        x, y = point[1], point[0]
        canvas.create_oval(x - point_size // 2, y - point_size // 2, x + point_size // 2, y + point_size // 2, fill=stroke_color.get(), outline=stroke_color.get())
        root.after(delay, draw_points, coords, index + 1)

if len(ray_points) > 0:
    draw_points(ray_points)

root.mainloop()
