import numpy as np
import paddle
from paddle import nn
from paddle.nn import Sequential, Conv2D, BatchNorm2D, ReLU, Dropout2D, AdaptiveMaxPool2D
import paddle.nn.functional as F
from resnet_dilated import ResNet50

class PSPModule(nn.Layer):
    def __init__(self, num_channels, size_list):
        super().__init__()
        self.size_list = size_list
        num_filters = num_channels // len(size_list)
        self.features_list = []
        for size in size_list:
            self.features_list.append(
                Sequential(
                    AdaptiveMaxPool2D(output_size=size),
                    Conv2D(in_channels=num_channels, out_channels=num_filters, kernel_size=1),
                    BatchNorm2D(num_filters),
                    ReLU()
                )
            )
        
    def forward(self, inputs):
        result = [inputs]
        for feature in self.features_list:
            x = feature(inputs)
            x = F.interpolate(x, inputs.shape[2:], mode='bilinear', align_corners=True)
            result.append(x)

        out = paddle.concat(result, axis=1)
        return out

class PSPNet(nn.Layer):
    def __init__(self, num_classes=59):
        super().__init__()

        res = ResNet50()

        self.layer0 = Sequential(
            res.conv,
            res.pool2d_max
        )
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4
        num_channels = 2048
        self.pspmodule = PSPModule(num_channels, [1, 2, 3, 6])
        num_channels *= 2

        self.classifier = Sequential(
            Conv2D(in_channels=num_channels, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2D(512),
            ReLU(),
            Dropout2D(0.1),
            Conv2D(in_channels=512, out_channels=num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pspmodule(x)
        x = self.classifier(x)
        x = F.interpolate(x, inputs.shape[2:], mode='bilinear', align_corners=True)

        return x

if __name__ == '__main__':
    x_data = np.random.rand(2, 3, 473, 473).astype(np.float32)
    x_input = paddle.to_tensor(x_data)
    model = PSPNet()
    model.eval()
    pred = model(x_input)
    print(pred.shape)
