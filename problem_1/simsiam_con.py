import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from utils.tool import get_conf, get_optimizer,LRScheduler,data_loading
from utils.model import SimSiam
from utils.knn import knn

parser = argparse.ArgumentParser(description='simsiam')
parser.add_argument('--lr', default=0.06, type=float)
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='BATCH')    
parser.add_argument('--epoches', default=150, type=int)
parser.add_argument('--wd', default=0.001, type=float, metavar='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--warmup_epoch', default=10, type=int)
parser.add_argument('--warmup_lr', default=0, type=float)
parser.add_argument('--final_lr', default=0, type=float)
parser.add_argument('--resize', default=32, type=int)
parser.add_argument('--load_path', default='./simsiam_con_results/train/best_loss.pkl', type=str,help='path to load the best model')      
parser.add_argument('--save_path', default='./simsiam_con_results/train', type=str,help='path to save the best model') 
parser.add_argument('--tensorboard_path', default='./simsiam_con_logs/train', type=str,help='path to log')   
args = parser.parse_args()

#设定随机种子
seed = 18
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = 256
num_epoch = args.epoches
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter(args.tensorboard_path)

mymodel=torch.load('./simsiam_results/train/last.pkl')

def train(loader, model, optimizer, scheduler, epoch):
    model.train()
    finish, total_loss = 0, 0.
    iterion = tqdm(loader)
    for (xi, xj), _ in iterion:
        xi, xj = xi.to(device), xj.to(device)

        if epoch == 1:
            grid = make_grid(xi[:32])
            writer.add_image('views_1', grid, global_step=epoch)

            grid = make_grid(xj[:32])
            writer.add_image('views_2', grid, global_step=epoch)

        model.zero_grad()
        loss = model(xi, xj)

        total_loss += loss.item()
        finish += 1

        loss.backward()
        optimizer.step()
        lr = scheduler.step()
        iterion.set_description('> Train_Epoch {}_Iter {}: loss {:.4f}'.format(
            epoch, finish, total_loss / finish
        ))
    return total_loss / finish, lr



train_loader,memory_loader,test_loader=data_loading(batch_size,model='simsiam',resize=args.resize)

model = SimSiam().to(device=device)
model.load_state_dict(mymodel['state_dict'])
optimizer = get_optimizer(model, args.lr, momentum=args.momentum, wd=args.wd)
optimizer.load_state_dict(mymodel['optimizer'])
scheduler=mymodel['scheduler']
begin_epoch=mymodel['epoch']

last_model_path = os.path.join(args.save_path, 'last.pkl')
best_loss_model_path = os.path.join(args.save_path, 'best_loss.pkl')
best_acc_model_path = os.path.join(args.save_path, 'best_acc.pkl')

best_loss = np.inf
best_acc = -1
for epoch in range(begin_epoch, num_epoch + 1):
    print('#' * 20, 'Epoch {}'.format(epoch), '#' * 20)
    train_loss, lr = train(train_loader, model, optimizer, scheduler, epoch)
    print('| Train_Epoch {}: loss {}, lr {}'.format(
        epoch, train_loss, lr
    ))
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('lr', lr, global_step=epoch)

    acc = knn(model.encoder.resnet, memory_loader, test_loader, epoch,
                        k=200, t=0.1, hide_progress=False, device=device)
    print('| Eval from kNN_Epoch {}: acc {:.4f}'.format(
        epoch, acc
    ))
    writer.add_scalar('knn_acc', acc, global_step=epoch)

    if best_loss > train_loss:
        best_loss = train_loss
        torch.save(model.encoder.resnet.state_dict(), best_loss_model_path)
    if best_acc < acc:
        best_acc = acc
        torch.save(model.encoder.resnet.state_dict(), best_acc_model_path)

    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'encoder': model.encoder.resnet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler,
    }, last_model_path)


