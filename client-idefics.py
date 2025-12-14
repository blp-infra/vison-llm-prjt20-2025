import grpc
import vision_pb2, vision_pb2_grpc

with open("image.jpg", "rb") as f:
    img_bytes = f.read()

channel = grpc.insecure_channel("localhost:50051")
stub = vision_pb2_grpc.VisionServiceStub(channel)

response = stub.Chat(
    vision_pb2.VisionRequest(
        image=img_bytes,
        prompt="Describe the image"
    )
)

print(response.text)
