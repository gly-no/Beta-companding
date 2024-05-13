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
from solve import W_Quan_betalaw, W_Quan_betalaw_new, W_Quan_normal, W_Quan_Kumaraswamy, W_learn_clip_beta_n2u, grad_a, A_Quan_betalaw_new
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def run(M):
    '''分布式训练初始化'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # torch.distributed.init_process_group(backend="nccl")
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)

    # device = torch.device("cuda", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##模型处理
    net = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    # print(net)

    '''运行之前记得修改保存文件！！'''
    epochs = 40
    batch_size = 512
    lr = 0.001
    # M = 2

    torch.set_grad_enabled(True)
    net.train()

    ##数据集处理

    train_data_path = "/data/dataset/Imagenet2012/train"
    val_data_path = "/data/dataset/Imagenet2012/val"
    # transform = transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_scale = 0.08
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    train_data = datasets.ImageFolder(root=train_data_path, transform=train_transforms)
    test_data = datasets.ImageFolder(root=val_data_path, transform=test_transforms)
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)
    # sampler_train = DistributedSampler(train_data) 
    # loader_train = DataLoader(train_data, batch_size=4*batch_size, sampler=sampler_train)
    # sampler_test = DistributedSampler(test_data) 
    # loader_test = DataLoader(test_data, batch_size=4*batch_size, sampler=sampler_test)

    '''gpu'''
    net.to(device)
    # net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    T1 = time.time()
  
    def act_hook(module,input,output):
        # print(f"{module} input: {input}")
        # print(f"{module} output: {output}")
        out = A_Quan_betalaw_new(output, M,name = 'A_1')
        output = out[0].to(device)
        # return out[0].to(device)

  
    # hook1 = net.layer1[0].relu.register_forward_hook(act_hook)
    # hook2 = net.layer1[1].relu.register_forward_hook(act_hook)
    hook3 = net.layer2[0].relu.register_forward_hook(act_hook)
    hook4 = net.layer2[1].relu.register_forward_hook(act_hook)
    hook5 = net.layer3[0].relu.register_forward_hook(act_hook)
    # hook6 = net.layer3[1].relu.register_forward_hook(act_hook)
    # hook7 = net.layer4[0].relu.register_forward_hook(act_hook)
    # hook8 = net.layer4[1].relu.register_forward_hook(act_hook)
    # hook9 = net.layer3[2].relu.register_forward_hook(act_hook)
    net.to(device)
    # net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    # net = net.module
    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch + 1))
        # net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        # loader_train.sampler.set_epoch(epoch)
        # loader_test.sampler.set_epoch(epoch)

        for i, data in enumerate(loader_train, 0):
            net.train()
            #prepare dataset
            length = len(loader_train)
            # print(length)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            #forward & backward
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #print ac & loss in each batch
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            if i % 100 == 99:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            hook1.remove()
            hook2.remove()
            hook3.remove()
            hook4.remove()
            hook5.remove()
            # hook6.remove()
            # hook7.remove()
            # hook8.remove()
        scheduler.step()
        if epoch != 0:
            hook1 = net.layer1[0].relu.register_forward_hook(act_hook)
            hook2 = net.layer1[1].relu.register_forward_hook(act_hook)
            hook3 = net.layer2[0].relu.register_forward_hook(act_hook)
            hook4 = net.layer2[1].relu.register_forward_hook(act_hook)
            hook5 = net.layer3[0].relu.register_forward_hook(act_hook)
            # hook6 = net.layer3[1].relu.register_forward_hook(act_hook)
            # hook7 = net.layer4[0].relu.register_forward_hook(act_hook)
            # hook8 = net.layer4[1].relu.register_forward_hook(act_hook)
    
            net.to(device)
            # net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
            # net = net.module
        if epoch > 1:
            print('Waiting Test...')
            with torch.no_grad():
                correct = 0
                total = 0
                for data in loader_test:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    hook1.remove()
                    hook2.remove()
                    hook3.remove()
                    hook4.remove()
                    hook5.remove()
                    # hook6.remove()
                    # hook7.remove()
                    # hook8.remove()
                print('Test\'s ac is: %.3f%%' % (100 * correct / total))
        

        
    print('Train has finished, total epoch is %d' % epochs)
    T2 = time.time()
    print('time:',(T2-T1)*1000,'ms')
    print('Finished Training')
    # torch.save(net, './VGG16_Quan_multi_conv_2bits.pkl')



if __name__ == "__main__":
    for M in range(2,6):
    # M=4
        print('M=',M)
        run(M)





