import os
import shutil

# Đường dẫn tới folder chứa các ảnh
src_folder = "Eval_dataset\CUFED5"
# Đường dẫn tới folder mới sẽ chứa các folder con
dst_root = "CUFED5"

# Tạo folder đích nếu chưa tồn tại
os.makedirs(dst_root, exist_ok=True)

# Duyệt qua tất cả các file trong folder gốc
for filename in os.listdir(src_folder):
    if filename.endswith(".png"):
        # Lấy prefix 3 ký tự đầu (ví dụ: "000" trong "000_0.png")
        album_id = filename[:3]

        # Tạo folder con tương ứng
        album_folder = os.path.join(dst_root, album_id)
        os.makedirs(album_folder, exist_ok=True)

        # Copy file vào folder con
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(album_folder, filename)
        shutil.copy2(src_path, dst_path)

print("Tách xong các album!")
