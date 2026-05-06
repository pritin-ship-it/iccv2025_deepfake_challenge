FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04


RUN apt-get update && apt-get install -y \
    python3-pip python3-dev build-essential \
    libgl1-mesa-glx libglib2.0-0 wget git && \
    rm -rf /var/lib/apt/lists/*
    
WORKDIR /workspace
COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3"]
#CMD ["--config_path", "configs/best_bboxes.py"]
