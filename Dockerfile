FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    g++

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

WORKDIR /project