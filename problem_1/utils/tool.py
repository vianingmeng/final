import yaml
import os
import torch
import numpy as np
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from utils.augment import TwoCropAugment, simsiam_transform, linear_transform

py_path = os.path.dirname(os.path.realpath(__file__))


def get_conf():
    conf_file = os.path.join(py_path, '../config.yaml')
    config = yaml.load(open(conf_file, 'r', encoding='UTF-8'), Loader=yaml.FullLoader)
    return config


def get_optimizer(model, lr, momentum, wd):
    predictor_prefix = 'predictor'
    parameters = [{
        'name': 'encoder',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    }, {
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
    optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd)
    # optimizer = optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optimizer

def data_loading(batch_size,aug=True,model='resnet18',resize=32):
    if model=='resnet18':
        # 准备数据
        if aug==True:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.ToTensor()
                    , transforms.RandomCrop(32, padding=4)  # 先四周填充0，在吧图像随机裁剪成32*32
                    , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
                    , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor()
                    , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
        else:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.ToTensor()
                    , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor()
                    , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }        
        
        image_datasets = {x: CIFAR10('./data', train=True if x == 'train' else False,
            transform=data_transforms[x], download=True) for x in ['train', 'test']}
    
        dataloaders: dict = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False
            ) for x in ['train', 'test']
        }
        return dataloaders
    if model=='simsiam':
        train_transform = simsiam_transform(resize=resize)
        test_transform = linear_transform()

        data_dir = './data'
        train_sets = CIFAR10(data_dir, train=True, download=True, transform=TwoCropAugment(train_transform))
        memory_sets = CIFAR10(data_dir, train=True, download=True, transform=test_transform)
        test_sets = CIFAR10(data_dir, train=False, download=True, transform=test_transform)

        train_loader = DataLoader(train_sets, batch_size=batch_size, shuffle=True)
        memory_loader = DataLoader(memory_sets, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_sets, batch_size=batch_size, shuffle=False)
        return train_loader,memory_loader,test_loader

class LRScheduler(object):

    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0

    def step(self):
        lr = None
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        return lr
