import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.tool import get_conf, get_optimizer,LRScheduler,data_loading
from utils.model import Linear
from utils.tool import LRScheduler

parser = argparse.ArgumentParser(description='finetune')
parser.add_argument('--lr', default=30, type=float)
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='BATCH')    
parser.add_argument('--epoches', default=50, type=int)
parser.add_argument('--wd', default=0, type=float, metavar='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--warmup_epoch', default=10, type=int)
parser.add_argument('--warmup_lr', default=0, type=float)
parser.add_argument('--final_lr', default=0, type=float)
parser.add_argument('--pretrained', default=False, type=bool,help='Pretrained or not') 
parser.add_argument('--load_path', default='./simsiam_results/train/best_loss.pkl', type=str,help='path to load the best model')      
parser.add_argument('--save_path', default='./simsiam_results/eval', type=str,help='path to save the best model') 
parser.add_argument('--tensorboard_path', default='./simsiam_logs/eval', type=str,help='path to log')   
args = parser.parse_args()

seed=18
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = args.batch_size
num_epoch = args.epoches
lr=args.lr
momentum=args.momentum
wd=args.wd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter(args.tensorboard_path)


def finetune(loader, model, criterion, optimizer, scheduler, epoch, is_train=True):
    model.train() if is_train else model.eval()

    iterion = tqdm(loader)
    finish, total_loss, total_num, total_acc = 0, 0., 0, 0.
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in iterion:
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            finish += 1
            total_loss += loss.item()
            total_num += target.shape[0]
            pred = torch.argsort(output, dim=-1, descending=True)
            total_acc += torch.sum((pred[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            if is_train:
                loss.backward()
                optimizer.step()
                scheduler.step()

            iterion.set_description('> {}_Epoch {}_Iter {}: loss {:.4f}, acc {:.4f}'.format(
                'Train' if is_train else 'Test', epoch, finish, total_loss / finish, total_acc / total_num
            ))
        return total_loss / finish, total_acc / total_num


def main():

    train_loader=data_loading(batch_size,aug=False)['train']
    test_loader=data_loading(batch_size,aug=False)['test']  

    model = Linear(out_dim=10).to(device)
    model.resnet.load_state_dict(torch.load(args.load_path))
    for layer in model.resnet.parameters():
        layer.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr, momentum=momentum, wd=wd)
    scheduler = LRScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epoch,
        warmup_lr=args.warmup_lr,
        num_epochs=num_epoch,
        base_lr=lr,
        final_lr=args.final_lr,
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=True
    )

    best_loss_model_path = os.path.join(args.save_path, 'best_loss.pkl')
    best_acc_model_path = os.path.join(args.save_path, 'best_acc.pkl')
    last_model_path = os.path.join(args.save_path, 'last.pkl')

    best_loss = np.inf
    best_acc = -1
    for epoch in range(1, num_epoch + 1):
        print('#' * 20, 'Epoch {}'.format(epoch), '#' * 20)
        train_loss, train_acc = finetune(train_loader, model, criterion, optimizer, scheduler, epoch)
        print('| Train_Epoch {}: loss {:.4f}, acc {:.4f}'.format(
            epoch, train_loss, train_acc
        ))
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('train_acc', train_acc, global_step=epoch)

        test_loss, test_acc = finetune(test_loader, model, criterion, optimizer, scheduler, epoch, is_train=False)
        print('| Test_Epoch {}: loss {:.4f}, acc {:.4f}'.format(
            epoch, test_loss, test_acc
        ))
        writer.add_scalar('test_loss', test_loss, global_step=epoch)
        writer.add_scalar('test_acc', test_acc, global_step=epoch)

        torch.save(model.state_dict(), last_model_path)
        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_loss_model_path)
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_acc_model_path)


if __name__ == '__main__':
    main()
