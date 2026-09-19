
import torch
import torch.nn as nn
from network.backbone.csnet_encoder import CSNetBlock

def test_csnet_block():
    print("Testing CSNetBlock...")

    # Case 1: expand_ratio = 1 (should have 5 layers, shuffle=False)
    print("Testing expand_ratio=1...")
    block1 = CSNetBlock(inp=32, oup=16, stride=1, dilation=1, expand_ratio=1, groups=2)
    x1 = torch.randn(1, 32, 64, 64)
    try:
        out1 = block1(x1)
        print("expand_ratio=1 pass: Success")
        print(f"Output shape: {out1.shape}")
    except IndexError as e:
        print(f"expand_ratio=1 pass: Failed with IndexError: {e}")
    except Exception as e:
        print(f"expand_ratio=1 pass: Failed with {e}")

    # Case 2: expand_ratio = 6 (should have 8 layers, shuffle=True)
    print("\nTesting expand_ratio=6...")
    block2 = CSNetBlock(inp=32, oup=32, stride=1, dilation=1, expand_ratio=6, groups=2)
    x2 = torch.randn(1, 32, 64, 64)
    try:
        out2 = block2(x2)
        print("expand_ratio=6 pass: Success")
        print(f"Output shape: {out2.shape}")
    except Exception as e:
        print(f"expand_ratio=6 pass: Failed with {e}")

if __name__ == "__main__":
    test_csnet_block()
