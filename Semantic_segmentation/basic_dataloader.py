import cv2
import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import DataLoader, Dataset
from paddle.vision.datasets import DatasetFolder

class basic_dataset(Dataset):
    def __init__(self, image_folder, image_file_list, transform):
        super.__init__()
        self.image_folder = image_folder
        self.image_file_list = image_file_list
        self.transfrom = transform
        self.file_list = self.read_list()

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.file_list)

    def read_list(self):
        data_list = []
        with open(self.image_file_list) as infile:
            for line in infile:
                data_path = os.path.join(self.image_folder,line.split()[0])
                label_path = os.path.join(self.image_folder, line.split()[1])
                data_list.append((data_path, label_path))
        return data_list