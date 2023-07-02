from torch.utils.tensorboard import SummaryWriter
import torch
writer = SummaryWriter("Trying_logs")     
resnet18 = torch.load('./resnet_results/data.pth')
resnet18_pretrained = torch.load('./resnet_results/data1.pth')

for i in range(len(resnet18['loss'])):
    writer.add_scalar("resnet18/loss",resnet18['loss'][i],i)
    writer.add_scalar("resnet18/acc",resnet18['test_acc'][i],i)

for i in range(len(resnet18_pretrained['loss'])):
    writer.add_scalar("resnet18_pretrained/loss",resnet18_pretrained['loss'][i],i)
    writer.add_scalar("resnet18_pretrained/acc",resnet18_pretrained['test_acc'][i],i)

writer.close() 
