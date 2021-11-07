import os
import paddle
from paddle.io import DataLoader
import paddle.optimizer as optim
from paddle.vision.transforms import Compose
from basic_transform import random_scale, random_horizontal_flip, custom_resize, custom_normalize, to_tensor
import argparse
from FCN8s import FCN8s
from basic_segloss import basic_seg_loss
from basic_dataset import basic_dataset
from utils import AverageMeter

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='basic')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--image_folder', type=str, default='./dummy_data')
parser.add_argument('--image_list_file', type=str, default='./dummy_data/list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=2)

args = parser.parse_args()


def train(dataloader, model, criterion, optimizer, epoch, total_batch):
    model.train()
    train_loss_meter = AverageMeter()
    for batch_id, data in enumerate(dataloader):
        batch_id += 1
        image = data[0]
        label = data[1]
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        n = image.shape[0]
        train_loss_meter.update(loss.numpy()[0], n)
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
              f"Step[{batch_id:04d}/{total_batch:04d}], " +
              f"Average Loss: {train_loss_meter.avg:4f}")

    return train_loss_meter.avg

def test(dataloader, model, criterion):
    train_loss_meter = AverageMeter()
    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]
        pred = model(image)
        loss = criterion(pred, label)
        n = image.shape[0]
        train_loss_meter.update(loss.numpy()[0], n)

    return train_loss_meter.avg
    

if __name__ == '__main__':
    device = paddle.device.get_device()
    paddle.device.set_device(device)
    basic_transforms = Compose(transforms=[
                                            random_horizontal_flip(),
                                            random_scale(),
                                            custom_resize(size=(256, 256), interpolation='nearest'),
                                            custom_normalize(data_format='HWC'),
                                            # grayscale(),
                                            to_tensor()])
    model = FCN8s(9)
    train_set = basic_dataset(image_folder='/home/aistudio/facade/images', label_folder='/home/aistudio/facade/labels_signle', image_file_list='/home/aistudio/facade/image_list.txt', transform=basic_transforms, usage='train')
    train_dataloader = DataLoader(dataset=train_set, shuffle=True, num_workers=0, batch_size=args.batch_size, use_shared_memory=False)
    
    test_set = basic_dataset(image_folder='/home/aistudio/facade/images', label_folder='/home/aistudio/facade/labels_signle', image_file_list='/home/aistudio/facade/image_list.txt', transform=basic_transforms, usage='test')
    test_dataloader = DataLoader(dataset=test_set, shuffle=False, num_workers=0, batch_size=1, use_shared_memory=False)

    total_batch = len(train_dataloader)
    criterion = basic_seg_loss
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=args.lr)
    for epoch in range(1, args.num_epochs+1):
        train_loss = train(train_dataloader,
                            model,
                            criterion,
                            optimizer,
                            epoch,
                            total_batch)
        print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss:.4f}")

        test_loss = test(test_dataloader,
                        model,
                        criterion)

        print(f"----- Epoch[{epoch}/{args.num_epochs}] Test Loss: {test_loss:.4f}")

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{train_loss}")

            # save model and optmizer states
            paddle.save(model.state_dict(), f'model.pdparams')
            paddle.save(optimizer.state_dict(), f'optimizer.pdopt')
            print(f'----- Save model: {model_path}.pdparams')
            print(f'----- Save optimizer: {model_path}.pdopt')

