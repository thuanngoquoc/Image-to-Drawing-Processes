import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDataHandler:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def upload_image(self, dataset_slug, image_path):
        local_dataset_folder = "data_folder"  # dùng đường dẫn tương đối
        metadata_path = os.path.join(local_dataset_folder, "dataset-metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("dataset-metadata.json không tồn tại trong data_folder")
    
        base_name = os.path.basename(image_path)
        dest_path = os.path.join(local_dataset_folder, base_name)
    
        if os.path.abspath(image_path) != os.path.abspath(dest_path):
            shutil.copyfile(image_path, dest_path)
        else:
            print("File ảnh đã ở trong data_folder, không cần copy.")
    
        self.api.dataset_create_version(
            local_dataset_folder,
            version_notes="Add new file",
            dir_mode="zip",
            quiet=False,
            convert_to_csv=False
        )
    
        
    