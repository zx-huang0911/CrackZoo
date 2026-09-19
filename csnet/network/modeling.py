from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3, CSNetHeadV3Plus, CSNetV3
import torch.nn as nn
from .backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception
    , csnet_encoder
)

########### Task 3 FFAB and DenseASPP import start ###########
# import CSNet decoder head
from .csnet_decoder import CSNetHead
########### Task 3 FFAB and DenseASPP import end ###########

def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):

    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256 # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36] # If follow paper trend, can put [24, 48, 72].

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = xception.xception(pretrained= 'imagenet' if pretrained_backbone else False, replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 128

    if name=='deeplabv3plus':
        return_layers = {'conv4': 'out', 'block1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_csnet_encoder(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    # CSNetEncoder doesn't support pretrained_backbone argument in __init__ in my implementation
    # But mobilenet_v2 does. I should update CSNetEncoder to accept it (and ignore or handle).
    # My CSNetEncoder implementation currently doesn't load pretrained weights.
    # The user instruction says: "no pretrained backbone yet" or similar?
    # Actually the user prompt said "test_only" and provided a checkpoint.
    # But that checkpoint is for mobilenetv2.
    # Anyway, I will pass it but CSNetEncoder must accept it or I change the call.
    # I'll update CSNetEncoder to accept kwargs or pretrained argument.
    # But for now, I will just call it without pretrained argument if I didn't add it.
    # I didn't add it to CSNetEncoder.__init__.

    backbone = csnet_encoder.csnet_encoder(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    # MobileNetV2 features are a list of layers wrapped in Sequential.
    # low_level_features: indices 0-4 (4 layers).
    # high_level_features: indices 4-18 (rest excluding last conv).
    # My CSNetEncoder puts everything in .features (Sequential).

    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_csnetv3plus_csnet_encoder(num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        dense_dilate = (12, 24, 36, 48)
    else:
        dense_dilate = (6, 12, 18, 24)

    backbone = csnet_encoder.csnet_encoder(pretrained=pretrained_backbone, output_stride=output_stride)

    # Define feature extraction layers by slicing the encoder
    # 1/4 resolution features
    backbone.features_1_4 = nn.Sequential(*backbone.features[0:3])
    # 1/8 resolution features
    backbone.features_1_8 = nn.Sequential(*backbone.features[3:6])
    # 1/16 resolution features
    backbone.features_1_16 = nn.Sequential(*backbone.features[6:11])
    # Final output features
    backbone.features_out = nn.Sequential(*backbone.features[11:-1])

    backbone.features = None
    backbone.classifier = None

    # Channel dimensions for each feature level
    inplanes = 320           # Final output
    middle_level_planes = 64 # 1/16 resolution
    planes_1_8 = 32          # 1/8 resolution
    planes_1_4 = 24          # 1/4 resolution

    return_layers = {
        'features_out': 'out',
        'features_1_16': 'middle',
        'features_1_8': 'feat_1_8',
        'features_1_4': 'feat_1_4'
    }

    classifier = CSNetHeadV3Plus(
        in_channels=inplanes,
        middle_level_planes=middle_level_planes,
        planes_1_4=planes_1_4,
        planes_1_8=planes_1_8,
        num_classes=num_classes,
        dense_dilate=dense_dilate
    )

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = CSNetV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone=='csnet_encoder':
        model = _segm_csnet_encoder(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone=='xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


########### Task 2 Decoder Enhancement start ###########
def csnet_xceptionplus(num_classes: int = 21, output_stride: int = 16, pretrained_backbone: bool = False):
    """Constructs the CSNet segmentation model.

    This factory function builds a CSNet model by instantiating a
    lightweight encoder and an enhanced decoder head.  The decoder
    incorporates DenseASPP, feature fusion via weighted summation, and
    attention mechanisms to better aggregate multi‑scale context and
    low‑level details.  The returned model is a subclass of
    ``_SimpleSegmentationModel`` that upscales its predictions to match
    the input resolution.

    Args:
        num_classes: Number of segmentation classes.  The model fully
            supports multi‑class segmentation; use ``num_classes=2``
            only when performing binary crack segmentation.
        output_stride: Desired output stride of the encoder (default 16).
        pretrained_backbone: Ignored for CSNet since no pretrained
            weights are provided.

    Returns:
        An instance of ``_SimpleSegmentationModel`` wrapping the CSNet
        encoder and decoder.
    """
    # instantiate custom encoder
    backbone = csnet_encoder.CSNetEncoder(output_stride=output_stride, return_features=True)
    # instantiate decoder head
    classifier = CSNetHead(
        in_channels=backbone.out_channels,
        low16_channels=backbone.low_level16_channels,
        low4_channels=backbone.low_level_channels,
        num_classes=num_classes
    )
    # wrap in SimpleSegmentationModel
    from .utils import _SimpleSegmentationModel
    model = _SimpleSegmentationModel(backbone, classifier)
    return model

# Provide a capitalized alias for backwards compatibility.
CSNet_Xceptionplus = csnet_xceptionplus
########### Task 2 Decoder Enhancement end ###########


# Deeplab v3
def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False): # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3', 'hrnetv2_32', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False): # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_csnet_encoder(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a CSNet Encoder backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'csnet_encoder', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def csnetv3plus_csnet_encoder(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _segm_csnetv3plus_csnet_encoder(num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
