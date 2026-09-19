
import sys
import os
import torch
import network.modeling as modeling

def main():
    print("Instantiating CSNet_Xceptionplus...")
    try:
        model = modeling.csnet_xceptionplus(num_classes=21, output_stride=16)
        print("Model instantiated successfully!")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return

    print("Running dummy forward pass...")
    try:
        input_tensor = torch.randn(1, 3, 513, 513)
        output = model(input_tensor)
        print(f"Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Failed forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
