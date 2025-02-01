import os
import time
import numpy as np

class SegmentProcessor:
    def run_segmentation(self, image_path):
        raise NotImplementedError()

class KaggleSegmentProcessor(SegmentProcessor):
    def __init__(self, kaggle_data_handler, dataset_slug):
        self.data_handler = kaggle_data_handler
        self.dataset_slug = dataset_slug

    def run_segmentation(self, image_path):
        time.sleep(5)

    
        # Tạo dữ liệu giả:
        mask_dummy = np.zeros((500,500), dtype=np.uint8)
        # vẽ 1 vòng tròn làm "head"
        import cv2
        cv2.circle(mask_dummy, (250,250), 50, 255, -1)
        
        objects = [
            {"class_name":"dog","id":16,"parts":{"head":mask_dummy.copy(),"ear":mask_dummy.copy()}},
            {"class_name":"car","id":2,"parts":{"frame":mask_dummy.copy(),"wheel":mask_dummy.copy()}},
            {"class_name":"cat","id":15,"parts":{"head":mask_dummy.copy(),"ear":mask_dummy.copy()}},
            {"class_name":"banana","id":46,"parts":{"whole":mask_dummy.copy()}}
        ]


