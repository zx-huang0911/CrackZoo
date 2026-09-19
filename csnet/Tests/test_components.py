
import unittest
import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.backbone.csnet_encoder import CSNetEncoder
from network._deeplab import CSNetHeadV3Plus, DenseASPP as DenseASPP_DeepLab
from network.csnet_decoder import CSNetHead, DenseASPP as DenseASPP_Decoder, FeatureFusionModule, AttentionModule
from utils.loss import GeneralizedDiceLoss

class TestCSNetComponents(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.num_classes = 21
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_csnet_encoder(self):
        print("\nTesting CSNetEncoder...")
        encoder = CSNetEncoder(output_stride=16, return_features=True).to(self.device)
        input_tensor = torch.randn(self.batch_size, 3, 513, 513).to(self.device)
        features = encoder(input_tensor)

        self.assertIsInstance(features, dict)
        self.assertIn('out', features)
        self.assertIn('low_level', features)
        self.assertIn('low_level16', features)

        print(f"Encoder Output Shapes: Out={features['out'].shape}, Low={features['low_level'].shape}, Low16={features['low_level16'].shape}")

        # Check channel attributes
        self.assertTrue(hasattr(encoder, 'out_channels'))
        self.assertTrue(hasattr(encoder, 'low_level_channels'))
        self.assertTrue(hasattr(encoder, 'low_level16_channels'))

    def test_dense_aspp_deeplab(self):
        print("\nTesting DenseASPP (from _deeplab.py)...")
        in_channels = 320
        daspp = DenseASPP_DeepLab(in_channels=in_channels, out_channels=256).to(self.device)
        input_tensor = torch.randn(self.batch_size, in_channels, 32, 32).to(self.device)
        output = daspp(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 256, 32, 32))

    def test_dense_aspp_decoder(self):
        print("\nTesting DenseASPP (from csnet_decoder.py)...")
        in_channels = 320
        daspp = DenseASPP_Decoder(in_channels=in_channels, out_channels=256).to(self.device)
        input_tensor = torch.randn(self.batch_size, in_channels, 32, 32).to(self.device)
        output = daspp(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 256, 32, 32))

    def test_feature_fusion_module(self):
        print("\nTesting FeatureFusionModule...")
        high_c, low_c, out_c = 64, 24, 64
        fusion = FeatureFusionModule(in_channels_high=high_c, in_channels_low=low_c, out_channels=out_c).to(self.device)
        high_feat = torch.randn(self.batch_size, high_c, 128, 128).to(self.device)
        low_feat = torch.randn(self.batch_size, low_c, 128, 128).to(self.device)
        output = fusion(high_feat, low_feat)
        self.assertEqual(output.shape, (self.batch_size, out_c, 128, 128))

    def test_attention_module(self):
        print("\nTesting AttentionModule...")
        ch = 64
        att = AttentionModule(in_channels=ch).to(self.device)
        input_tensor = torch.randn(self.batch_size, ch, 128, 128).to(self.device)
        output = att(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, ch, 128, 128))

    def test_csnet_head_v3plus(self):
        print("\nTesting CSNetHeadV3Plus (used in csnetv3plus_csnet_encoder)...")
        # Based on _segm_csnetv3plus_csnet_encoder logic:
        # inplanes = 320 (out)
        # middle_level_planes = 64 (1/16)
        # planes_1_8 = 32
        # planes_1_4 = 24

        head = CSNetHeadV3Plus(
            in_channels=320,
            middle_level_planes=64,
            planes_1_4=24,
            planes_1_8=32,
            num_classes=self.num_classes
        ).to(self.device)

        # Mock features from IntermediateLayerGetter
        features = {
            'out': torch.randn(self.batch_size, 320, 32, 32).to(self.device),       # 1/16 of 513 is ~32
            'middle': torch.randn(self.batch_size, 64, 32, 32).to(self.device),     # 1/16
            'feat_1_8': torch.randn(self.batch_size, 32, 64, 64).to(self.device),   # 1/8
            'feat_1_4': torch.randn(self.batch_size, 24, 128, 128).to(self.device)  # 1/4
        }

        output = head(features)
        print(f"CSNetHeadV3Plus Output Shape: {output.shape}")
        # Expected output spatial size should match the largest input (feat_1_4) -> 128x128
        # Because the head upsamples to 1/4 resolution.
        # The final DeepLabV3+ model further upsamples by 4 to get 513.
        self.assertEqual(output.shape, (self.batch_size, self.num_classes, 128, 128))

    def test_generalized_dice_loss(self):
        print("\nTesting GeneralizedDiceLoss...")
        loss_fn = GeneralizedDiceLoss(ignore_index=255)
        logits = torch.randn(self.batch_size, self.num_classes, 32, 32).to(self.device)
        targets = torch.randint(0, self.num_classes, (self.batch_size, 32, 32)).to(self.device)
        loss = loss_fn(logits, targets)
        print(f"Dice Loss: {loss.item()}")
        self.assertTrue(loss.item() >= 0)

if __name__ == '__main__':
    unittest.main()
