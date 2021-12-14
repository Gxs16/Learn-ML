from functools import partial

from torch import nn
from torch.nn import Module, Hardswish, ReLU, Hardsigmoid, Sequential, BatchNorm2d, AdaptiveAvgPool2d, Conv2d
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
from torchinfo import summary

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class bneck(Module):
    def __init__(self, in_channels, out_channels, expand_size, kernel_size, is_se, nonlinearity, stride, dilation):
        super().__init__()
        layers = []
        if nonlinearity == 'HS':
            activation_layer = Hardswish
        elif nonlinearity == 'RE':
            activation_layer = ReLU

        self.use_res_connect = stride == 1 and in_channels == out_channels
        norm_layer = partial(BatchNorm2d, eps=0.001, momentum=0.01)
        # Expand
        if expand_size != in_channels:
            layers.append(ConvNormActivation(in_channels, expand_size, kernel_size=1, norm_layer=norm_layer,activation_layer=activation_layer))
        # Depthwise
        layers.append(ConvNormActivation(expand_size, expand_size, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=expand_size, norm_layer=norm_layer, activation_layer=activation_layer))
        # Squeeze-and-Excite
        if is_se:
            layers.append(SqueezeExcitation(expand_size, _make_divisible(expand_size//4), scale_activation=Hardsigmoid))
        # project
        layers.append(ConvNormActivation(expand_size, out_channels, kernel_size=1, activation_layer=None, norm_layer=norm_layer))

        self.block = Sequential(*layers)

    def forward(self, inputs):
        result = self.block(inputs)
        if self.use_res_connect:
            result += inputs

        return result

class MobileNetV3(nn.Module):
    def __init__(self, config_dict_list, last_channel, num_classes):
        super().__init__()
        layers = []

        norm_layer = partial(BatchNorm2d, eps=0.001, momentum=0.01)
        # First layer
        layers.append(ConvNormActivation(3, config_dict_list[0]['in_channels'], kernel_size=3, stride=2, activation_layer=Hardswish, norm_layer=norm_layer))
        # Blocks
        for config in config_dict_list:
            layers.append(bneck(**config))

        # last conv
        lastconv_output_channels = config_dict_list[-1]['out_channels']*6
        layers.append(ConvNormActivation(config_dict_list[-1]['out_channels'], lastconv_output_channels, kernel_size=1, norm_layer=norm_layer))
        self.features = Sequential(*layers)

        self.avgpool = AdaptiveAvgPool2d(1)
        self.classifier = Sequential(
            Conv2d(lastconv_output_channels, last_channel, 1),
            Hardswish(),
            Conv2d(last_channel, num_classes, 1)
        )

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

mobilenetv3_small_config = [
    {'in_channels':16, 'out_channels':16, 'expand_size':16, 'kernel_size':3, 'is_se':True, 'nonlinearity':'RE', 'stride':2, 'dilation':1},
    {'in_channels':16, 'out_channels':24, 'expand_size':72, 'kernel_size':3, 'is_se':False, 'nonlinearity':'RE', 'stride':2, 'dilation':1},
    {'in_channels':24, 'out_channels':24, 'expand_size':88, 'kernel_size':3, 'is_se':False, 'nonlinearity':'RE', 'stride':1, 'dilation':1},
    {'in_channels':24, 'out_channels':40, 'expand_size':96, 'kernel_size':5, 'is_se':True, 'nonlinearity':'HS', 'stride':2, 'dilation':1},
    {'in_channels':40, 'out_channels':40, 'expand_size':240, 'kernel_size':5, 'is_se':True, 'nonlinearity':'HS', 'stride':1, 'dilation':1},
    {'in_channels':40, 'out_channels':40, 'expand_size':240, 'kernel_size':5, 'is_se':True, 'nonlinearity':'HS', 'stride':1, 'dilation':1},
    {'in_channels':40, 'out_channels':48, 'expand_size':120, 'kernel_size':5, 'is_se':True, 'nonlinearity':'HS', 'stride':1, 'dilation':1},
    {'in_channels':48, 'out_channels':48, 'expand_size':144, 'kernel_size':5, 'is_se':True, 'nonlinearity':'HS', 'stride':1, 'dilation':1},
    {'in_channels':48, 'out_channels':96, 'expand_size':288, 'kernel_size':5, 'is_se':True, 'nonlinearity':'HS', 'stride':2, 'dilation':1},
    {'in_channels':96, 'out_channels':96, 'expand_size':576, 'kernel_size':5, 'is_se':True, 'nonlinearity':'HS', 'stride':1, 'dilation':1},
    {'in_channels':96, 'out_channels':96, 'expand_size':576, 'kernel_size':5, 'is_se':True, 'nonlinearity':'HS', 'stride':1, 'dilation':1},
]

if __name__ == '__main__':
    net = MobileNetV3(mobilenetv3_small_config, num_classes=1000, last_channel=1024)
    print(summary(net, input_size=(1, 3, 224, 224)))


