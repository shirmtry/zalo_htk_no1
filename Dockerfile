# Dùng môi trường PyTorch có sẵn GPU
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip uninstall -y torchaudio torchvision

# Copy toàn bộ code và model vào trong Docker
COPY . .

# Đảm bảo predict.sh thực thi được
RUN chmod +x /code/predict.sh

# Lệnh chạy khi Docker khởi động
CMD ["/code/predict.sh"]