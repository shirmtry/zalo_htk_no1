import os
from huggingface_hub import hf_hub_download

# 1. Kích hoạt hf_transfer để tăng tốc độ tải (nếu chưa cài thì chạy: pip install hf_transfer)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 2. Cấu hình thông tin
REPO_ID = "Huy1223/zalo_submit_htk"
FILENAME = "zalo_dataspace_v5.tar"
# Dán token của bạn vào đây (ví dụ: "hf_abc123xyz...")
HF_TOKEN = "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" 

# 3. Tải file
print(f"⏳ Đang tải {FILENAME} từ {REPO_ID}...")
try:
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",  # <-- DÒNG QUAN TRỌNG NHẤT BỊ THIẾU LÚC NÃY
        token=HF_TOKEN,       # Xác thực để tránh bị giới hạn tốc độ
        local_dir="."         # Lưu file vào thư mục hiện tại đang chạy script
    )
    print(f"✅ Tải thành công! File được lưu tại: {file_path}")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")