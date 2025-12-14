# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# WORKDIR /app

# RUN apt update && apt install -y python3 python3-pip git
# RUN pip install torch transformers accelerate pillow fastapi uvicorn

# # Pre-download model
# RUN python3 - <<EOF
# from transformers import AutoProcessor, AutoModelForVision2Seq
# AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
# AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b")
# EOF

# COPY main.py .

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

#---------gRPC Image-multistage--------
# 
# Stage 1: build
FROM python:3.12 AS builder
WORKDIR /app
RUN pip install grpcio-tools
COPY vision.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vision.proto

# Stage 2: runtime
FROM nvidia/cuda:12.8.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y python3 python3-pip
RUN pip install torch transformers pillow grpcio 

# Pre-download model
RUN python3 - <<EOF
from transformers import AutoProcessor, AutoModelForVision2Seq
AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b")
EOF

COPY --from=builder /app/*.py .
COPY server.py .
CMD ["python3", "server.py"]

# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# WORKDIR /app

# RUN apt update && apt install -y python3 python3-pip
# RUN pip install torch transformers pillow grpcio grpcio-tools

# # Pre-download model
# RUN python3 - <<EOF
# from transformers import AutoProcessor, AutoModelForVision2Seq
# AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
# AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b")
# EOF

# COPY . .

# CMD ["python3", "server.py"]
