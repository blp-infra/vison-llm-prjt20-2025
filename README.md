# Vision-Language Model Inference Service (gRPC + GPU)

A production-ready, high-throughput **Vision-Language Model (VLM) inference service** built using **IDEFICS-2 (8B)**, exposed via **gRPC** and optimized for **GPU-based local deployment**.

This project demonstrates how to deploy large multimodal models as **scalable, containerized microservices** without relying on external inference APIs.

---

## 🚀 Features

- 🧠 **Multimodal inference** (image + text)
- ⚡ **gRPC-based API** for low-latency and high throughput
- 🖥️ **GPU-accelerated inference** using PyTorch (FP16)
- 📦 **Dockerized deployment** using NVIDIA CUDA Runtime images
- 🔒 **Offline inference** (model downloaded once, no internet required at runtime)
- 👥 Designed for **multi-client concurrent access**
- 🛠️ Clean separation of build-time and runtime dependencies

---

## 🏗️ Architecture
```
Clients (Python / Go / Java / Rust)
|
| gRPC
v
Vision Inference Service
|
| Local GPU Inference
v
IDEFICS-2 Vision-Language Model

```
---

## 📦 Tech Stack

- **Language:** Python
- **Model:** HuggingFace IDEFICS-2 (8B)
- **Frameworks:** PyTorch, Transformers
- **API:** gRPC (Protocol Buffers)
- **Containerization:** Docker
- **GPU:** NVIDIA CUDA + cuDNN
- **OS:** Linux (Ubuntu)

---

## ⚙️ Setup & Build

### 1️⃣ Generate gRPC stubs
```bash
python -m grpc_tools.protoc \
  -I. \
  --python_out=. \
  --grpc_python_out=. \
  vision.proto

2️⃣ Build Docker image
```
docker build -t idefics-grpc:tag .
```
3️⃣ Run with GPU
```
docker run --gpus all -p 50051:50051 idefics-grpc
```
📡 gRPC API
```
Service

service VisionService {
  rpc Chat (VisionRequest) returns (VisionResponse);
}

Request

    Image (bytes)

    Prompt (string)

Response

    Generated text output
```

🧪 Example Client (Python)
```
import grpc
import vision_pb2, vision_pb2_grpc

with open("image.jpg", "rb") as f:
    img = f.read()

channel = grpc.insecure_channel("localhost:50051")
stub = vision_pb2_grpc.VisionServiceStub(channel)

response = stub.Chat(
    vision_pb2.VisionRequest(
        image=img,
        prompt="Describe the image"
    )
)

print(response.text)
```

📈 Performance Notes

    Optimized for batching and concurrent requests

    Ideal for internal AI platforms and private deployments

    Supports deployment on:

        RTX 3090 / 4090

        A100 / L4 / L40S

🔐 Security & Access

    No Hugging Face tokens required at runtime

    Model runs fully locally

    API authentication can be added (API key / mTLS) as needed

📌 Use Cases

    Enterprise multimodal AI services

    On-prem AI inference

    Internal developer platforms (IDP)

    Vision-based automation systems

🧠 Future Enhancements

    Dynamic request batching

    Streaming token responses

    Authentication & rate limiting

    Kubernetes deployment with GPU autoscaling

    Triton / Ray Serve integration

📜 License

This project is intended for educational and internal deployment use.
Please review the model license before commercial usage.
