import os
from paddle.io import Dataset
from paddle.vision import image_load

class basic_dataset(Dataset):
    def __init__(self, image_folder, label_folder, image_file_list, transform=None, usage='train'):
        super().__init__()
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_file_list = image_file_list
        self.transform = transform
        self.usage = usage
        self.file_list = self.read_list()

    def __getitem__(self, idx):
        data_path, label_path = self.file_list[idx]
        data = image_load(data_path, backend='cv2')
        label = image_load(label_path, backend='cv2').astype('float32')
        if self.transform:
            data, label = self.transform((data, label))
        return data, label

    def __len__(self):
        return len(self.file_list)

    def read_list(self):
        data_list = []
        with open(self.image_file_list) as infile:
            for line in infile:
                data_path = os.path.join(self.image_folder, line.split()[0])
                label_path = os.path.join(self.label_folder, line.split()[0].replace('jpg', 'png'))
                data_list.append((data_path, label_path))
        length = int(len(data_list)*0.3)
        if self.usage == 'train':
            return data_list[length:]
        elif self.usage == 'test':
            return data_list[:length]