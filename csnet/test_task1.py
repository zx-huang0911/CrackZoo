import torch
import sys
import os

# Add current directory to path so we can import network
sys.path.append(os.getcwd())
from network.backbone.csnet_encoder import csnet_encoder
from network.modeling import deeplabv3plus_csnet_encoder

def test_encoder():
    print("Testing CSNetEncoder...")
    # Use output_stride=16 for standard DeepLabV3+ config
    model = csnet_encoder(output_stride=16, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 513, 513)

    # Note: CSNetEncoder.forward returns the final features (Sequential forward)
    # But modeling.py slices it manually.

    # Check low level features access (simulating what modeling.py does)
    low_level = model.features[0:4]
    high_level = model.features[4:-1]

    out_low = low_level(x)
    print(f"Low Level Feature shape: {out_low.shape}")

    out_high = high_level(out_low)
    print(f"High Level Feature shape: {out_high.shape}")

    # Check if dimensions match expectations for DeepLabV3+
    # MobileNetV2:
    # Low level (after 4 layers, i.e., index 0,1,2,3):
    # 0: Conv3x3 (s=2) -> 257
    # 1: Bottle(t=1,c=16,n=1,s=1) -> 257
    # 2: Bottle(t=6,c=24,n=2,s=2) -> 129 (stride 2 here)
    # 3: Bottle(t=6,c=24,n=2,s=1) -> 129
    # Output channels: 24.

    if out_low.shape[1] == 24 and out_low.shape[2] == 129:
        print("Encoder low level dimensions check passed.")
    else:
        print(f"Encoder low level dimensions check FAILED: {out_low.shape}")

    # High level (end): Stride 16.
    # Output channels: 320.
    if out_high.shape[1] == 320: # and out_high.shape[2] == 33 (513/16 = 32.06 -> 33)
        print("Encoder high level dimensions check passed.")
    else:
        print(f"Encoder high level dimensions check FAILED: {out_high.shape}")

def test_model():
    print("\nTesting Full Model (deeplabv3plus_csnet_encoder)...")
    try:
        model = deeplabv3plus_csnet_encoder(num_classes=21, output_stride=16)
        model.eval()
        x = torch.randn(2, 3, 513, 513)
        with torch.no_grad():
            out = model(x)
        print(f"Model Output shape: {out.shape}")
        if out.shape == (2, 21, 513, 513):
            print("Model forward pass passed.")
        else:
            print("Model forward pass FAILED (shape mismatch).")
    except Exception as e:
        print(f"Model forward pass FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_encoder()
    test_model()
