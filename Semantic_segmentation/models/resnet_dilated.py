import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm2D, MaxPool2D, AdaptiveAvgPool2D,Linear, ReLU, Sequential

class ConvBNLayer(nn.Layer):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, act=None, dilation=1, padding=None):
        super().__init__()

        if padding:
            padding = padding
        else:
            padding = (kernel_size-1)//2

        self.act = act
        self.conv = Conv2D(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           bias_attr=False)
        self.relu = ReLU()

        self.bn = BatchNorm2D(num_features=out_channels)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.act == 'relu':
            x = self.relu(x)
        return x

class BasicBlock(nn.Layer): # 
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, shortcut=True):
        super().__init__()
        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 act='relu')

        self.conv1 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 act=None)

        self.relu = ReLU()

        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     stride=stride,
                                     act=None)

        self.shortcut = shortcut

    def forward(self, inputs):
        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)
        if self.shortcuts:
            short = inputs
        else:
            short = self.short(inputs)
        y = short+conv1
        y = self.relu(y)
        return y

class BottleneckBlock(nn.Layer): # 
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, shortcut=True,
                 dilation=1, padding=None):
        super().__init__()
        self.num_channel_out = out_channels*4
        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 act='relu')
        self.conv1 = ConvBNLayer(in_channels=out_channels, 
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 padding=padding,
                                 act='relu',
                                 dilation=dilation)
        self.conv2 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=self.num_channel_out,
                                 kernel_size=1,
                                 stride=1)
        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=self.num_channel_out,
                                     kernel_size=1, stride=stride)

        self.relu = ReLU()
        self.shortcut = shortcut
        
    def forward(self, inputs):
        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = short+conv2
        y = self.relu(y)
        return y

class ResNet(nn.Layer):
    def __init__(self, layers=50, num_classes=1000):
        super().__init__()
        self.layers = layers

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34:
            depth = [3, 4, 6, 3]
        elif layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        else:
            raise Exception('supported layers: [18, 34, 50, 101, 152]')

        if layers < 50:
            in_channels = [64, 64, 128, 256]
            block = BasicBlock
            l1_shortcut = True
        else:
            in_channels = [64, 256, 512, 1024]
            block = BottleneckBlock
            l1_shortcut = False

        out_channels = [64, 128, 256, 512]

        self.conv = ConvBNLayer(in_channels=3,
                                out_channels=64,
                                kernel_size=7,
                                stride=2,
                                act='relu')

        self.pool2d_max = MaxPool2D(kernel_size=3,
                                    stride=2,
                                    padding=1)

        self.layer1 = Sequential(
            *self._make_layer(block=block,
                                in_channels=in_channels[0],
                                out_channels=out_channels[0],
                                depth=depth[0],
                                stride=1,
                                shortcut=l1_shortcut)
        )
        self.layer2 = Sequential(
            *self._make_layer(block=block,
                                in_channels=in_channels[1],
                                out_channels=out_channels[1],
                                depth=depth[1],
                                stride=2,
                                shortcut=l1_shortcut,
                                dilation=2)
        )
        self.layer3 = Sequential(
            *self._make_layer(block=block,
                                in_channels=in_channels[2],
                                out_channels=out_channels[2],
                                depth=depth[2],
                                stride=1,
                                shortcut=l1_shortcut,
                                dilation=2)
        )
        self.layer4 = Sequential(
            *self._make_layer(block=block,
                                in_channels=in_channels[3],
                                out_channels=out_channels[3],
                                depth=depth[3],
                                stride=1,
                                shortcut=l1_shortcut,
                                dilation=4)
        )
        self.last_pool = AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc = Linear(in_features=out_channels[-1]*block.expansion,
                            out_features=num_classes)

    def _make_layer(self, block, in_channels, out_channels, depth, stride, dilation=1, shortcut=False):
        layers = []
        if dilation > 1:
            padding = dilation
        else:
            padding = None

        layers.append(block(in_channels,
                            out_channels,
                            stride=stride,
                            shortcut=shortcut,
                            dilation=dilation,
                            padding=padding))

        for i in range(1, depth):
            layers.append(block(out_channels*block.expansion,
                                out_channels,
                                stride=1,
                                dilation=dilation,
                                padding=padding))
        return layers

    def forward(self, inputs):
        x = self.conv(inputs)
        print(x.shape)
        x = self.pool2d_max(x)

        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        x = self.last_pool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x

def ResNet50():
        return ResNet()

    

if __name__ == '__main__':
    
    x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
    x = paddle.to_tensor(x_data)
    model = ResNet50()
    model.eval()
    pred = model(x)
    print('dilated resnet50: pred.shape = ', pred.shape)