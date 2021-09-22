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
    def __init__(self, num_classes=11):
        super(LeNet, self).__init__()
        # [3, 128, 128]
        self.conv1 = Conv2D(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        # [64, 128, 128]
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # [64, 64, 64]
        self.conv2 = Conv2D(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        # [128, 64, 64]
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # [128, 32, 32]
        # 创建第3个卷积层
        self.conv3 = Conv2D(in_channels=128, out_channels=512, kernel_size=5, padding=2)
        # [512, 32, 32]
        self.max_pool3 = MaxPool2D(kernel_size=2, stride=2)
        # [512, 16, 16]

        self.conv4 = Conv2D(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        # [512, 16, 16]
        self.max_pool4 = MaxPool2D(kernel_size=2, stride=2)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        self.fc1 = Linear(in_features=512*8*8, out_features=512)
        self.fc2 = Linear(in_features=512, out_features=128)
        
        self.fc3 = Linear(in_features=128, out_features=num_classes)

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
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
# %%
epoch_num = 30
learning_rate = 0.001

model = LeNet()
loss = paddle.nn.loss.CrossEntropyLoss() # 因为是分类任务，所以 loss 使用 CrossEntropyLoss
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters()) # optimizer 使用 Adam

#%%
print('start training...')
for epoch in range(epoch_num):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # 模型训练
    model.train()
    for img, label in train_loader():
        optimizer.clear_grad()
        pred = model(img)
        step_loss = loss(pred, label)
        step_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(pred.numpy(), axis=1) == label.numpy())
        train_loss += step_loss.numpy()[0]

    # 模型验证
    model.eval()
    for img, label in val_loader():
        pred = model(img)
        step_loss = loss(pred, label)
        
        val_acc += np.sum(np.argmax(pred.numpy(), axis=1) == label.numpy())
        val_loss += step_loss.numpy()[0]

    # 将结果打印出来
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, epoch_num, \
                 time.time()-epoch_start_time, \
                 train_acc/traindataset.__len__(), \
                 train_loss/traindataset.__len__(), \
                 val_acc/valdataset.__len__(), \
                 val_loss/valdataset.__len__()))