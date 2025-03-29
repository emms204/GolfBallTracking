FROM debian:bullseye-slim

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    python3 \
    python3-pip \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /tmp/
# RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Create app directory
WORKDIR /app

# Copy project files
COPY . /app/

# Download and extract ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.0.tgz \
    && rm onnxruntime-linux-x64-1.16.0.tgz \
    && mv onnxruntime-linux-x64-1.16.0 onnxruntime-linux-x64-1.21.0

# Build the project
RUN cd src \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make

# Set the library path
ENV LD_LIBRARY_PATH=/app/onnxruntime-linux-x64-1.21.0/lib:$LD_LIBRARY_PATH

# Create an entrypoint script
RUN echo '#!/bin/bash \n\
if [ "$1" = "camera" ]; then \n\
    ./onnx_detector --camera=0 --model=/app/best.onnx --classes=/app/classes.txt --conf=0.25 --nms=0.45 \n\
elif [ "$1" = "video" ] && [ -n "$2" ]; then \n\
    ./onnx_detector --video="/app/videos/$2" --model=/app/best.onnx --classes=/app/classes.txt --conf=0.25 --nms=0.45 \n\
else \n\
    echo "Usage:" \n\
    echo "  For camera input: docker run -it --rm --device=/dev/video0 myapp camera" \n\
    echo "  For video input: docker run -it --rm -v /path/to/videos:/app/videos myapp video your_video.mp4" \n\
fi' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Make a symlink to the executable
RUN ln -s /app/src/build/onnx_detector /app/onnx_detector

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (will show usage if no arguments provided)
CMD ["help"] 