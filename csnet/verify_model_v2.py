
import sys
import os
import torch
import network.modeling as modeling

def main():
    print("Instantiating csnetv3plus_csnet_encoder...")
    try:
        # User uses: python main.py --model csnetv3plus_csnet_encoder --output_stride 16 ...
        # Based on network/modeling.py, the function name might be different from the string passed to dict.
        # Let's check modeling.py again to see how the string maps to the function.
        # But assuming the user command works, there must be a mapping.
        # I will try to find it in modeling.py first or just call the likely function.
        # _segm_csnetv3plus_csnet_encoder seems to be the internal one.
        # I need to find the public factory function.

        # Checking modeling.py via import
        if hasattr(modeling, 'csnetv3plus_csnet_encoder'):
             model = modeling.csnetv3plus_csnet_encoder(num_classes=21, output_stride=16)
        else:
             print("Function csnetv3plus_csnet_encoder not found in modeling.py directly.")
             # It might be registered in a dict or accessible via a different name.
             # Let's try to construct it manually if needed or look for it.
             return

        print("Model instantiated successfully!")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
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
