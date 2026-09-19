import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3", "CSNetV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.

        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.

        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class CSNetV3(_SimpleSegmentationModel):
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DenseASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=(6, 12, 18, 24), inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(out_channels // (len(dilations) + 2), 1)

        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        in_ch = in_channels + inter_channels
        for dilation in dilations:
            stage = nn.Sequential(
                nn.Conv2d(in_ch, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
            )
            self.stages.append(stage)
            in_ch += inter_channels

        self.pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        concat_channels = in_channels + inter_channels * (len(dilations) + 2)
        self.project = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = [x]

        point = self.point_conv(x)
        features.append(point)

        for stage in self.stages:
            inp = torch.cat(features, dim=1)
            out = stage(inp)
            features.append(out)

        pool = self.pool(x)
        pool = F.interpolate(pool, size=size, mode='bilinear', align_corners=False)
        features.append(pool)

        concat = torch.cat(features, dim=1)
        return self.project(concat)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_channels = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CSNetHeadV3Plus(nn.Module):
    def __init__(self, in_channels, middle_level_planes, planes_1_4, planes_1_8, num_classes, dense_dilate=(6, 12, 18, 24)):
        super().__init__()
        # Learnable weights for feature fusion
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

        # 1x1 conv to unify channel dimensions before summation
        self.conv_1_4 = nn.Conv2d(planes_1_4, planes_1_8, 1, bias=False)
        self.bn_1_4 = nn.BatchNorm2d(planes_1_8)

        self.cbam = CBAM(planes_1_8)

        self.dense_aspp = DenseASPP(in_channels, out_channels=256, dilations=dense_dilate)

        # Fusion of middle_level (1/16) and aspp output
        self.fuse_middle = nn.Sequential(
            nn.Conv2d(256 + middle_level_planes, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # The input channels for the final classifier will be 256 (from fused_feature) + planes_1_8 (after fusion)
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + planes_1_8, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self._init_weight()

    def forward(self, feature):
        # Feature Fusion Module
        feat_1_4 = feature['feat_1_4']
        feat_1_8 = feature['feat_1_8']

        # Unify channel dimension of 1/4 features to match 1/8 features
        feat_1_4_unified = self.bn_1_4(self.conv_1_4(feat_1_4))

        # Upsample 1/8 features to match 1/4 features spatial dimensions
        feat_1_8_upsampled = F.interpolate(feat_1_8, size=feat_1_4.shape[2:], mode='bilinear', align_corners=False)

        # Element-wise weighted summation
        low_level_feature = self.w1 * feat_1_4_unified + self.w2 * feat_1_8_upsampled

        # CBAM
        low_level_feature = self.cbam(low_level_feature)

        # Process high-level features with DenseASPP
        aspp_feature = self.dense_aspp(feature['out'])

        # Fuse with middle-level features (1/16)
        middle_feature = feature['middle']
        # Upsample aspp_feature to match middle_feature size
        aspp_feature_upsampled = F.interpolate(aspp_feature, size=middle_feature.shape[2:], mode='bilinear', align_corners=False)
        fused_feature = torch.cat([aspp_feature_upsampled, middle_feature], dim=1)
        fused_feature = self.fuse_middle(fused_feature)

        # Upsample and concatenate with the new low_level_feature
        output_feature = F.interpolate(fused_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)

        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels,
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
