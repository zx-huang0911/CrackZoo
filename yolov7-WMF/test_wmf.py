
import torch
from models.yolo import Model
import yaml

def test():
    cfg = 'cfg/training/yolov7-wmf.yaml'
    device = torch.device('cpu')

    print(f"Loading model from {cfg}...")
    try:
        model = Model(cfg, ch=3, nc=80).to(device)
        print(model)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

    print("Running forward pass with dummy input...")
    dummy_input = torch.rand(1, 3, 640, 640).to(device)
    try:
        output = model(dummy_input)
        print("Forward pass successful.")
        print(f"Output shape: {[o.shape for o in output[0]]}") # Training output: [list of tensors]
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

if __name__ == '__main__':
    test()
