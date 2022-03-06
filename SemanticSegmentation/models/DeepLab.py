import numpy as np
import paddle
from paddle.nn import Layer, Conv2D, BatchNorm2D, Sequential, ReLU, AdaptiveMaxPool2D
import paddle.nn.functional as F
from resnet_multi_grid import ResNet50

class ASPPPooling(Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.features = Sequential(
            Conv2D(in_channels, out_channels, 1),
            BatchNorm2D(out_channels),
            ReLU()
        )
        self.adapt_pool = AdaptiveMaxPool2D(1)

    def forward(self, inputs):
        
        x = self.adapt_pool(inputs)
        x = self.features(x)
        x = F.interpolate(x, inputs.shape[2:], mode='bilinear')
        return x


class ASPPConv(Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            Conv2D(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            BatchNorm2D(out_channels),
            ReLU()
        )

class ASPPModule(Layer):
    def __init__(self, in_channels, out_channels, rates_list):
        super().__init__()
        self.features = [
            Sequential(
                Conv2D(in_channels, out_channels, 1),
                BatchNorm2D(out_channels),
                ReLU()
            ),
            ASPPPooling(in_channels, out_channels)
        ]
        for rate in rates_list:
            self.features.append(
                ASPPConv(in_channels, out_channels, rate)
            )
        self.project = Sequential(
            Conv2D(out_channels*(2+len(rates_list)), out_channels, 1),
            BatchNorm2D(out_channels),
            ReLU()
        )

    def forward(self, inputs):
        result = []
        for operation in self.features:
            result.append(operation(inputs))

        x = paddle.concat(result, axis=1)
        x = self.project(x)
        return x



    
class DeepLabHead(Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__(
                    ASPPModule(in_channels, 256, [12, 24, 36]),
                    Conv2D(256, 256, 3, padding=1),
                    BatchNorm2D(256),
                    ReLU(),
                    Conv2D(256, num_classes, 1)
        )


class DeepLab(Layer):
    def __init__(self, num_classes=59):
        super().__init__()
        resnet = ResNet50()

        self.layer0 = Sequential(
            resnet.conv,
            resnet.pool2d_max
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.layer5 = resnet.layer5
        self.layer6 = resnet.layer6
        self.layer7 = resnet.layer7

        feature_dim = 2048
        self.classifier = DeepLabHead(feature_dim, num_classes)

    def forward(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.classifier(x)
        x = F.interpolate(x, inputs.shape[2:], mode='bilinear', align_corners=True)

        return x

if __name__ == '__main__':
    x_data = np.random.rand(2, 3, 473, 473).astype(np.float32)
    x_input = paddle.to_tensor(x_data)
    model = DeepLab()
    model.eval()
    pred = model(x_input)
    print(pred.shape)
