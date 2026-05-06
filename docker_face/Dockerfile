FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04


RUN apt-get update && apt-get install -y \
    python3-pip python3-dev build-essential \
    libgl1-mesa-glx libglib2.0-0 wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .


RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt
RUN pip3 install --no-cache-dir scikit-learn

ENTRYPOINT ["python3"]
#CMD ["--config_path", "configs/best_bboxes.py"]
