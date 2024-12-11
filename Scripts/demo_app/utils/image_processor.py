import cv2

class ImageProcessor:
    def to_sketch(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # edges hiện tại: nét trắng (255) trên nền đen (0)
        
        # Đảo ngược màu để có nét đen trên nền trắng
        inverted = cv2.bitwise_not(edges)
        
        # Chuyển sang BGR để đồng nhất với định dạng ảnh gốc
        sketch = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
        return sketch

    def get_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
