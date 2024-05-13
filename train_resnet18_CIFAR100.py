import torch
import torchvision
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
from torchvision import datasets, models, transforms
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from solve import W_Quan_betalaw, W_Quan_betalaw_new, W_Quan_normal, W_Quan_Kumaraswamy
import os
import pickle

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # device = "cpu"
    ##模型处理
    net = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    # print(net)
    # net.classifier[0] = nn.Linear(in_features=25088, out_features=512, bias=True)
    # net.classifier[3] = nn.Linear(in_features=512, out_features=512, bias=True)
    # net.classifier[6] = nn.Linear(in_features=512, out_features=10, bias=True)
    net.fc = nn.Linear(in_features=512, out_features=100, bias=True)
    # torch.nn.init.normal_(net.classifier[6].weight.data, mean=0.0, std=1.0)
    # print(net)
    # print(net.classifier)
    
    '''gpu'''
    net.to(device)
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # net = net.cuda()
    # net = torch.nn.parallel.DistributedDataParallel(net)
    # net = torch.nn.DataParallel(net)

    '''运行之前记得修改保存文件！！'''
    epoch = 30
    batch_size = 64
    lr = 0.0001
    M = 8

    torch.set_grad_enabled(True)#在接下来的计算中每一次运算产生的节点都是可以求导的
    net.train()

    ##数据集
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data= datasets.CIFAR100(root='/data/dataset',train=True,transform=transform,download=True)
    train_loader = Data.DataLoader(train_data,batch_size = batch_size,shuffle=True,num_workers=2,drop_last=True)
    test_data = torchvision.datasets.CIFAR100(root='/data/dataset',train=False,transform=transform,download=True)
    test_loader = Data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=2,drop_last=True)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    torch.cuda.synchronize()
    T1 = time.time()
    for epoch in range(epoch):
        running_loss = 0
        running_corrects = 0

        for step, data in enumerate(train_loader):
            torch.set_grad_enabled(True)#在接下来的计算中每一次运算产生的节点都是可以求导的
            net.train()
           
            net.to(device)
            input, label = data
            optimizer.zero_grad()

            output = net(input.to(device))            
            loss = loss_function(output, label.to(device))
            loss.backward()

            
            running_loss += loss.item()
            pred = output.data.argmax(dim=1)
            running_corrects += torch.sum(pred.cpu() == label.data)
            if step % 190 == 189:
                print('Epoch', epoch+1,',step', step+1,'|Loss_avg:',running_loss/(step+1),'|Acc_avg:',running_corrects/((step+1)*batch_size))
                '''test acc'''

                torch.set_grad_enabled(False)
                net.eval()
                acc = 0.0
                for i, data in enumerate(test_loader):
                    x, y = data
                    y_pred = net(x.to(device, torch.float))
                    pred_t = y_pred.argmax(dim=1)
                    acc += (pred_t.data == y.data.to(device)).sum()
                acc = (acc / 10000) * 100
                print('Accuracy: %.2f' %acc, '%')

            ##更新
            optimizer.step()
        # running_loss = 0
        scheduler.step()
    torch.cuda.synchronize()
    T2 = time.time()
    print('time:',(T2-T1)*1000,'ms')
    print('Finished Training')
    torch.save(net.state_dict(), './RESNET18_CIFAR100_0.0001.pkl')
    
