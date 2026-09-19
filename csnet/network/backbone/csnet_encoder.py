from torch import nn
try: # for torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except: # for torchvision>=0.4
    from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

__all__ = ['CSNetEncoder', 'csnet_encoder']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        #padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, 0, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

def fixed_padding(kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)

########### Task 1 Encoder Enhancement start ###########
class CSNetBlock(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(CSNetBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

        self.input_padding = fixed_padding( 3, dilation )

    def forward(self, x):
        x_pad = F.pad(x, self.input_padding)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)

class CSNetEncoder(nn.Module):
    def __init__(self, num_classes=1000, output_stride=8, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, return_features=False):
        """
        CSNet Encoder (MobileNet V2 based)
        """
        super(CSNetEncoder, self).__init__()
        self.return_features = return_features
        block = CSNetBlock
        input_channel = 32
        last_channel = 1280
        self.output_stride = output_stride
        current_stride = 1
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        current_stride *= 2
        dilation=1
        previous_dilation = 1

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            previous_dilation = dilation
            if current_stride == output_stride:
                stride = 1
                dilation *= s
            else:
                stride = s
                current_stride *= s
            output_channel = int(c * width_mult)

            for i in range(n):
                if i==0:
                    features.append(block(input_channel, output_channel, stride, previous_dilation, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, dilation, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # Define channels for segmentation (CSNet)
        def get_channels(module):
            if isinstance(module, ConvBNReLU):
                return module[0].out_channels
            elif isinstance(module, CSNetBlock):
                return module.conv[-2].out_channels
            return 0

        # 1/4 resolution feature (index 3)
        self.low_level_channels = get_channels(self.features[3])
        # 1/16 resolution feature (index 13)
        self.low_level16_channels = get_channels(self.features[13])
        # Deepest feature before 1x1 expansion (index 17)
        self.out_channels = get_channels(self.features[17])

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.return_features:
            low_level_features = None
            low_level16_features = None
            out_features = None

            for i, module in enumerate(self.features):
                x = module(x)
                if i == 3:
                    low_level_features = x
                elif i == 13:
                    low_level16_features = x
                elif i == 17:
                    out_features = x

            return {
                'out': out_features,
                'low_level': low_level_features,
                'low_level16': low_level16_features
            }

        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

def csnet_encoder(pretrained=False, progress=True, **kwargs):
    """
    Constructs a CSNet Encoder (MobileNetV2-based) architecture.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = CSNetEncoder(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
########### Task 1 Encoder Enhancement end ###########
