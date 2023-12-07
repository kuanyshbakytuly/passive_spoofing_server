FROM python:3.10


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /project