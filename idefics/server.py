# import requests
# import torch
# from PIL import Image
# from io import BytesIO

# from transformers import AutoProcessor, AutoModelForVision2Seq
# from transformers.image_utils import load_image

# DEVICE = "cuda:0"

# # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
# image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
# image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
# image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

# processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
# model = AutoModelForVision2Seq.from_pretrained(
#     "HuggingFaceM4/idefics2-8b",
# ).to(DEVICE)

# # Create inputs
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "What do we see in this image?"},
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [
#             {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
#         ]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "And how about this image?"},
#         ]
#     },       
# ]
# prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
# inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# # Generate
# generated_ids = model.generate(**inputs, max_new_tokens=500)
# generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

# print(generated_texts)
# 
# ############ gRPC############
import grpc
from concurrent import futures
import vision_pb2, vision_pb2_grpc

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from io import BytesIO

DEVICE = "cuda:0"

# Load once (IMPORTANT)
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(DEVICE).eval()

class VisionService(vision_pb2_grpc.VisionServiceServicer):

    def Chat(self, request, context):
        image = Image.open(BytesIO(request.image)).convert("RGB")
        prompt = request.prompt

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]

        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )

        text = processor.decode(
            outputs[0], skip_special_tokens=True
        )

        return vision_pb2.VisionResponse(text=text)


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ],
    )

    vision_pb2_grpc.add_VisionServiceServicer_to_server(
        VisionService(), server
    )

    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC Vision Server running on :50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
