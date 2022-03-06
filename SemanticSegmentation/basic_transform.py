import random
import numpy as np
import paddle
from paddle.vision.transforms import BaseTransform
import paddle.vision.transforms.functional as F
from paddle.vision.transforms.transforms import Normalize, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor, Grayscale

def _get_image_size(img):

    return img.shape[:2][::-1]

class random_scale(BaseTransform):
    '''
    随机缩放
    '''
    def __init__(self, equal=True, min_scale=0.5, max_scale=2.0, keys=('image', 'label')):
        '''
        params:

        equal: 长与宽是否等距离缩放；

        min_scale: 最小缩放比例；

        max_scale: 最大缩放比例；

        keys：需要处理的数据格式；
        '''
        super().__init__(keys)
        self.min_scale = min_scale
        self.max_scale = max_scale
        if equal:
            _scale = np.random.uniform(low=self.min_scale, high=self.max_scale)
            self.scale = (_scale, _scale)
        else:
            self.scale = (np.random.uniform(low=self.min_scale, high=self.max_scale),
                               np.random.uniform(low=self.min_scale, high=self.max_scale))

    def _apply_image(self, image):
        height, width, _c = image.shape
        scale = self.scale
        image = F.resize(image, (int(width*scale[0]), int(height*scale[1])), interpolation='nearest')
        return image

    def _apply_label(self, label):
        height, width, _c = label.shape
        scale = self.scale
        label = F.resize(label, (int(width*scale[0]), int(height*scale[1])), interpolation='nearest')
        return label

class random_horizontal_flip(RandomHorizontalFlip):
    def __init__(self, prob=0.5, keys=('image', 'label')):
        super().__init__(prob=prob, keys=keys)
        self.random = random.random()

    def _apply_image(self, img):
        if self.random < self.prob:
            return F.vflip(img)
        return img
    
    def _apply_label(self, label):
        if self.random < self.prob:
            return F.vflip(label)
        return label

class random_vertical_flip(RandomVerticalFlip):
    def __init__(self, prob=0.5, keys=('image', 'label')):
        super().__init__(prob=prob, keys=keys)
        self.random = random.random()

    def _apply_image(self, img):
        if self.random < self.prob:
            return F.hflip(img)
        return img
    
    def _apply_label(self, label):
        if self.random < self.prob:
            return F.hflip(label)
        return label

class custom_resize(Resize):
    def __init__(self, size, interpolation='bilinear', keys=('image', 'label')):
        super().__init__(size=size, interpolation=interpolation, keys=keys)

    def _apply_label(self, label):
        return super()._apply_image(label)

class custom_normalize(Normalize):
    def __init__(self, mean=127.5,
                 std=127.5,
                 data_format='CHW',
                 to_rgb=False,
                 keys=('image', 'label')):
        super().__init__(mean=mean, std=std, data_format=data_format, to_rgb=to_rgb, keys=keys)

    def _apply_label(self, label):
        return label

    def _apply_image(self, image):
        return super()._apply_image(image)

class to_tensor(ToTensor):
    def __init__(self, data_format='CHW', keys=('image', 'label')):
        super().__init__(data_format=data_format, keys=keys)

    def _apply_label(self, label):
        return super()._apply_image(label)

class grayscale(Grayscale):
    def __init__(self, num_output_channels=1, keys=('image', 'label')):
        super().__init__(num_output_channels=num_output_channels, keys=keys)

    def _apply_image(self, image):
        return image

    def _apply_label(self, label):
        return super()._apply_image(label)

# class CustomNormalize(Normalize):

if __name__ == '__main__':

    img = (np.random.rand(100, 120, 3) * 255.).astype(np.float32)

    # # test RandomScale
    # trans = RandomScale()
    # outputs = trans(img)
    # print(outputs.shape)
    # outputs = trans((img, img))
    # print(outputs[0].shape, outputs[1].shape)
    

    # # test CustomRandomHorizontalFlip
    # trans = CustomRandomHorizontalFlip()
    # outputs = trans(img)
    # print(outputs.shape)
    # outputs = trans((img, img))
    # print(outputs[0].shape, outputs[1].shape)

    # # test CustomRandomVerticalFlip
    # trans = CustomRandomVerticalFlip()
    # outputs = trans(img)
    # print(outputs.shape)
    # outputs = trans((img, img))
    # print(outputs[0].shape, outputs[1].shape)

    # # test Normalize
    # print(img)
    # trans = Normalize(mean=[127.5, 127.5, 127.5],
    #                     std=[127.5, 127.5, 127.5],
    #                     data_format='HWC')
    # outputs = trans(img)
    # print(outputs.shape)
    # outputs = trans((img, img))
    # print(outputs[0].shape, outputs[1].shape)
    # print(outputs[1])
    
    print('test finish!')
