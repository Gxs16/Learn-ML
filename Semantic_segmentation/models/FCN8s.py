import numpy as np
import paddle
from paddle.nn import Sequential, Conv2D, BatchNorm2D, ReLU, MaxPool2D, Conv2DTranspose, Dropout

class FCN8s(paddle.nn.Layer):
    def __init__(self, num_classes=59):
        super().__init__()
        self.layer_block1 = Sequential(
            Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=100),
            BatchNorm2D(num_features=64),
            ReLU(),
            Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=64),
            ReLU(),
            MaxPool2D(kernel_size=2, ceil_mode=True)
        )
        self.layer_block2 = Sequential(
            Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=128),
            ReLU(),
            Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=128),
            ReLU(),
            MaxPool2D(kernel_size=2, ceil_mode=True)
        )
        self.layer_block3 = Sequential(
            Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=256),
            ReLU(),
            Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=256),
            ReLU(),
            Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=256),
            ReLU(),
            MaxPool2D(kernel_size=2, ceil_mode=True)
        )
        self.layer_block4 = Sequential(
            Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=512),
            ReLU(),
            Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=512),
            ReLU(),
            Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=512),
            ReLU(),
            MaxPool2D(kernel_size=2, ceil_mode=True)
        )
        self.layer_block5 = Sequential(
            Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=512),
            ReLU(),
            Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=512),
            ReLU(),
            Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=512),
            ReLU(),
            MaxPool2D(kernel_size=2, ceil_mode=True)
        )
        self.layer_block6 = Sequential(
            Conv2D(in_channels=512, out_channels=4096, kernel_size=7),
            ReLU()
        )
        self.layer_block7 = Sequential(
            Conv2D(in_channels=4096, out_channels=4096, kernel_size=1),
            ReLU()
        )
        self.score = Conv2D(4096, num_classes, 1)
        self.score_pool3 = Conv2D(256, num_classes, 1)
        self.score_pool4 = Conv2D(512, num_classes, 1)
        self.drop6 = Dropout()
        self.drop7 = Dropout()

        self.up_output = Conv2DTranspose(in_channels=num_classes,
                                         out_channels=num_classes,
                                         kernel_size=4, stride=2, bias_attr=False)

        self.up_pool4 = Conv2DTranspose(in_channels=num_classes,
                                         out_channels=num_classes,
                                         kernel_size=4, stride=2, bias_attr=False)

        self.up_final = Conv2DTranspose(in_channels=num_classes,
                                         out_channels=num_classes,
                                         kernel_size=16, stride=8, bias_attr=False)
    
    def forward(self, input):
        x = self.layer_block1(input)
        x = self.layer_block2(x)
        x = self.layer_block3(x)
        pool3 = x
        x = self.layer_block4(x)
        pool4 = x
        x = self.layer_block5(x)
        x = self.layer_block6(x)
        x = self.drop6(x)
        x = self.layer_block7(x)
        x = self.drop7(x)

        x = self.score(x)
        x = self.up_output(x)
        up_output = x
        x = self.score_pool4(pool4)
        x = x[:, :, 5:5+up_output.shape[2], 5:5+up_output.shape[3]]
        up_pool4 = x
        x = up_pool4 + up_output
        x = self.up_pool4(x)
        up_pool4 = x

        x = self.score_pool3(pool3)
        x = x[:, :, 9:9+up_pool4.shape[2], 9:9+up_pool4.shape[3]]
        up_pool3 = x
        x = up_pool3 + up_pool4

        x = self.up_final(x)
        x = x[:, :, 31:31+input.shape[2], 31:31+input.shape[3]]

        return x

if __name__ == '__main__':
    x_data = np.random.rand(2, 3, 128, 128).astype(np.float32)
    x = paddle.to_tensor(x_data)
    model = FCN8s(num_classes=59)
    model.eval()
    pred = model(x)
    print(pred.shape)
