# 导入需要的包
import os
import cv2
import time
import numpy as np
import paddle
import paddle
import numpy as np
from paddle.io import Dataset, DataLoader
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.vision.transforms import Compose, Transpose, RandomRotation, RandomHorizontalFlip, Normalize, Resize
# 分配GPU设备
place = paddle.CUDAPlace(0)
paddle.disable_static(place)
## 组网
import paddle.nn.functional as F

# 定义 LeNet 网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        # 创建卷积和池化层
        # 创建第1个卷积层
        self.conv1 = Conv2D(in_channels=3, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 尺寸的逻辑：池化层未改变通道数；当前通道数为6
        # 创建第2个卷积层
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 创建第3个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 输入size是[28,28]，经过三次卷积和两次池化之后，C*H*W等于120
        self.fc1 = Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)
    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

class FoodDataset(Dataset):
    def __init__(self, image_path, image_size=(128, 128), mode='train'):
        self.image_path = image_path
        self.image_file_list = sorted(os.listdir(image_path))
        self.mode = mode
        # training 时做 data augmentation
        self.train_transforms = Compose([
            Resize(size=image_size),
            RandomHorizontalFlip(),
            RandomRotation(15),
            Transpose(),
            Normalize(mean=127.5, std=127.5)
        ])
        # testing 时不需做 data augmentation
        self.test_transforms = Compose([
            Resize(size=image_size),
            Transpose(),
            Normalize(mean=127.5, std=127.5)
        ])
        
    def __len__(self):
        return len(self.image_file_list)
    
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_path, self.image_file_list[idx]))
        if self.mode == 'train':
            img = self.train_transforms(img)
            label = int(self.image_file_list[idx].split("_")[0])
            return img, label
        else:
            img = self.test_transforms(img)
            return img

batch_size = 128
traindataset = FoodDataset('work/food-11/training')
valdataset = FoodDataset('work/food-11/validation')

train_loader = DataLoader(traindataset, places=paddle.CUDAPlace(0), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(valdataset, places=paddle.CUDAPlace(0), batch_size=batch_size, shuffle=False, drop_last=True)
