python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vision.proto

docker run --gpus all -p 50051:50051 idefics-grpc
