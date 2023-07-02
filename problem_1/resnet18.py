import argparse
import time
import copy
import numpy as np
import torch
import torchvision.models
from tqdm import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from utils.tool import data_loading
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Which model you choose')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR')
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N')    
    parser.add_argument('--epoches', default=130, type=int, metavar='N')
    parser.add_argument('--pretrained', default=False, type=bool,help='Pretrained or not') 
    parser.add_argument('--save_path', default='./resnet_results/', type=str,help='path to store the best model')      

    args = parser.parse_args()




    writer = SummaryWriter('/kaggle/working/logs')
    # GPU计算
    device = torch.device("cuda")
    batch_size = args.batch_size
    Lr = args.lr
    SAVE_PATH=args.save_path

    if args.pretrained==False:
        total_epochs = args.epoches
        filename = '{}best_resnet_model'.format(SAVE_PATH) 
    else:
        total_epochs = args.epoches-80
        filename = '{}best_pretrained_model'.format(SAVE_PATH)                  

    # SAVE_PATH = '/kaggle/working/'

 
    #提高速度，优化运行效率
    torch.backends.cudnn.benchmark = True

    # 加载数据集
    dataloaders= data_loading(batch_size)

    # 放入Resnet18架构
    if args.pretrained==False:
        model = torchvision.models.resnet18(pretrained=False)
    else:
        model = torchvision.models.resnet18(pretrained=True)
 
    # 修改模型
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    num_ftrs = model.fc.in_features  # 获取（fc）层的输入的特征数
    model.fc = nn.Linear(num_ftrs, 10)
 
    model.to(device)

 
    # 训练模型
    # 显示要训练的模型
    print("==============当前模型要训练的层==============")
    for name, params in model.named_parameters():
        if params.requires_grad:
            print(name)
 
    # 训练模型所需参数
    # 用于记录损失值未发生变化batch数
    counter = 0
    # 记录训练次数
    total_step = {
        'train': 0, 'test': 0
    }
    # 记录开始时间
    since = time.time()
    # 记录当前最小损失值
    valid_loss_min = np.Inf
    # 保存模型文件的尾标
    save_num = 0
    # 保存最优正确率
    best_acc = 0

    total_loss = []
    test_acc = []

    # 创建损失函数
    Loss = nn.CrossEntropyLoss()
    Loss.to(device)
 
    for epoch in range(total_epochs):
        # 动态调整学习率
        if counter / 10 == 1:
            counter = 0
            Lr = Lr * 0.5
 
        
        optimizer = optim.SGD(model.parameters(), lr=Lr, momentum=0.9, weight_decay=5e-4)
 
        print('Epoch {}/{}'.format(epoch + 1, total_epochs))
        print('-' * 10)
        print()
        # 训练和验证 每一轮都是先训练train 再test
        for phase in ['train', 'test']:
            # 调整模型状态
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证
 
            # 记录损失值
            running_loss = 0.0
            # 记录正确个数
            running_corrects = 0
 
            # 一次读取一个batch里面的全部数据
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                # 梯度清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = Loss(outputs, labels)
 
                    # torch.max() 返回的是一个元组 第一个参数是返回的最大值的数值 第二个参数是最大值的序号
                    _, preds = torch.max(outputs, 1)  # 前向传播 这里可以测试 在valid时梯度是否变化
 
                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()  # 反向传播
                        optimizer.step()  # 优化权重
 
                # 计算损失值
                running_loss += loss.item() * inputs.size(0)  # loss计算的是平均值，所以要乘上batch-size，计算损失的总和
                running_corrects += (preds == labels).sum()  # 计算预测正确总个数
                # 每个batch加1次
                total_step[phase] += 1
 
            # 一轮训练完后计算损失率和正确率
            if phase == 'train':
                epoch_loss = running_loss / len(dataloaders[phase].sampler)  # 当前轮的总体平均损失值
                epoch_acc = float(running_corrects) / len(dataloaders[phase].sampler)  # 当前轮的总正确率
                writer.add_scalar('train_loss', epoch_loss, global_step=epoch)    
                writer.add_scalar('train_acc', epoch_acc, global_step=epoch)
                total_loss.append(epoch_loss)
            else:
                epoch_loss = running_loss / len(dataloaders[phase].sampler)  # 当前轮的总体平均损失值
                epoch_acc = float(running_corrects) / len(dataloaders[phase].sampler)  # 当前轮的总正确率                
                writer.add_scalar('test_loss', epoch_loss, global_step=epoch)    
                writer.add_scalar('test_acc', epoch_acc, global_step=epoch)
                test_acc.append(epoch_acc)

            time_elapsed = time.time() - since
            print()
            print('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format(phase, epoch_loss, counter, epoch_acc))
 
            if phase == 'test':
                # 得到最好那次的模型
                if epoch_loss < valid_loss_min:  # epoch_acc > best_acc:
 
                    best_acc = epoch_acc
 
                    # 保存当前模型
                    best_model_wts = copy.deepcopy(model.state_dict())
                    state = {
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                    }
                    # 只保存最近2次的训练结果
                    save_num = 0 if save_num > 1 else save_num
                    save_name_t = '{}_{}.pth'.format(filename, save_num)
                    torch.save(state, save_name_t) 
                    print("已保存最优模型，准确率:\033[1;31m {:.2f}%\033[0m，文件名：{}".format(best_acc * 100, save_name_t))
                    save_num += 1
                    valid_loss_min = epoch_loss
                    counter = 0
                else:
                    counter += 1
 
        print()
        print('当前学习率 : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        print()
 
    # 训练结束
    time_elapsed = time.time() - since
    print()
    print('任务完成！')
    print('任务完成总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('最高准确率: {:4f}'.format(best_acc))
    save_num = save_num - 1
    save_num = save_num if save_num < 0 else 1
    save_name_t = '{}_{}.pth'.format(filename, save_num)
    print('最优模型保存在：{}'.format(save_name_t))
    data = {'loss':total_loss,'test_acc':test_acc}
    #torch.save(data,'/kaggle/working/data.pth')
    if args.pretrained==False:
        torch.save(data,'{}data.pth'.format(SAVE_PATH))
    else:
        torch.save(data,'{}pretrained_data.pth'.format(SAVE_PATH))