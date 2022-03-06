import numpy as np
import paddle as paddle
from paddle import nn
from paddle.nn import Conv2D, BatchNorm2D, ReLU, MaxPool2D, Conv2DTranspose
import paddle.nn.functional as F

class Encoder(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()

        self.pool = MaxPool2D(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x_pooled = self.pool(x)

        return x, x_pooled

class Decoder(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = Conv2DTranspose(in_channels, out_channels, kernel_size=2, stride=2)

        
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()

    def forward(self, inputs_prev, inputs):
        x = self.up(inputs)
        h_diff = inputs_prev.shape[2] - x.shape[2]
        w_diff = inputs_prev.shape[3] - x.shape[3]
        x = F.pad(x, pad=[h_diff//2, h_diff-h_diff//2, w_diff//2, w_diff-w_diff//2])
        x = paddle.concat([inputs_prev, x], axis=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x
        

class UNet(nn.Layer):
    def __init__(self, num_classes=59):
        super().__init__()
        self.down1 = Encoder(in_channels=3, out_channels=64)
        self.down2 = Encoder(in_channels=64, out_channels=128)
        self.down3 = Encoder(in_channels=128, out_channels=256)
        self.down4 = Encoder(in_channels=256, out_channels=512)

        self.mid_conv1 = Conv2D(512, 1024, kernel_size=1)
        self.mid_bn1 = BatchNorm2D(1024)
        self.mid_relu1 = ReLU()
        self.mid_conv2 = Conv2D(1024, 1024, kernel_size=1)
        self.mid_bn2 = BatchNorm2D(1024)
        self.mid_relu2 = ReLU()

        self.up4 = Decoder(in_channels=1024, out_channels=512)
        self.up3 = Decoder(in_channels=512, out_channels=256)
        self.up2 = Decoder(in_channels=256, out_channels=128)
        self.up1 = Decoder(in_channels=128, out_channels=64)

        self.last_conv = Conv2D(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, inputs):
        x1, x = self.down1(inputs)
        # print(x1.shape, x.shape)
        x2, x = self.down2(x)
        # print(x2.shape, x.shape)
        x3, x = self.down3(x)
        # print(x3.shape, x.shape)
        x4, x = self.down4(x)
        # print(x4.shape, x.shape)

        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

        # print(x4.shape, x.shape)
        x = self.up4(x4, x)
        # print(x3.shape, x.shape)
        x = self.up3(x3, x)
        # print(x2.shape, x.shape)
        x = self.up2(x2, x)
        # print(x1.shape, x.shape)
        x = self.up1(x1, x)
        # print(x.shape)

        x = self.last_conv(x)

        return x

if __name__ == '__main__':
    x_data = np.random.rand(1, 3, 572, 572).astype(np.float32)
    x_input = paddle.to_tensor(x_data)
    model = UNet(num_classes=2)
    model.eval()
    pred = model(x_input)
    print(pred.shape)
