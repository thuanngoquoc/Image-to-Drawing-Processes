import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Đảm bảo rằng file kaggle.json được lưu ở thư mục cấu hình đúng
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')  # Đảm bảo đúng đường dẫn tới thư mục .kaggle

# Khởi tạo Kaggle API
api = KaggleApi()
api.authenticate()  # Xác thực API

# Lấy danh sách các dataset trên Kaggle
datasets = api.datasets_list()

# In thông tin của mỗi dataset
for dataset in datasets:
    print(f"Dataset Name: {dataset['ref']}")  # Sử dụng 'ref' từ dict thay vì attribute

