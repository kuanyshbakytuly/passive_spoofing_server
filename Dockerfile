FROM python:3.10


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

WORKDIR /project