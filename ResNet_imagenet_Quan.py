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
from solve import W_Quan_betalaw, W_Quan_betalaw_new, W_Quan_normal, W_Quan_Kumaraswamy, W_learn_clip_beta_n2u, grad_a
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json




def run(M):
    '''分布式训练初始化'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
    # torch.distributed.init_process_group(backend="nccl")
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)

    # device = torch.device("cuda", local_rank)


    ##模型处理
    net = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    

    # w_c1 = net.conv1.weight.data.reshape(1,-1)[0].numpy()
    w_c2 = net.layer2[1].conv2.weight.data.reshape(1,-1)[0].numpy()
    # w_c3 = net.layer4[1].conv2.weight.data.reshape(1,-1)[0].numpy()
    # w_c4 = net.fc.weight.data.reshape(1,-1)[0].numpy()
  

    hist, edge = np.histogram(w_c2, bins=100, density=True)
    av_edge = [0]*100
    for i in range(100):
        av_edge[i] = (edge[i+1] + edge[i]) / 2
    out_hist = dict(zip(av_edge,hist))
    
    #保存
    with open('latex_plot/resnet18_fc_hist.json', 'w') as f:
        json.dump(out_hist, f)
  
    # plt.clf()
    
    # plt.hist(w_c1, bins= 100, alpha = 1, label= '第一层',density= True)
    # plt.hist(w_c2, bins= 100, alpha = 1, label= '第二层',density= True)
    # plt.hist(w_c3, bins= 100, alpha = 1, label= '第三层',density= True)
    # plt.hist(w_c4, bins= 100, alpha = 1, label= '第四层',density= True)
    # # plt.hist(w_c5, bins= 100, alpha = 1, label= '第五层',density= True)
 
    # plt.xticks(fontsize=12,weight='bold')
    # plt.yticks(fontsize=12,weight='bold')  
    # plt.rcParams.update({'font.size': 12})
    # plt.legend(loc = 'upper right', prop = {'weight':"bold"})
    # plt.show()
    # plt.savefig("plot/resnet18_weight_distri.png")


    '''运行之前记得修改保存文件！！'''
    epochs = 40
    batch_size = 200
    lr = 0.0001
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

    sampler_train = DistributedSampler(train_data) 
    loader_train = DataLoader(train_data, batch_size=4*batch_size, sampler=sampler_train)
    sampler_test = DistributedSampler(test_data) 
    loader_test = DataLoader(test_data, batch_size=4*batch_size, sampler=sampler_test)

    '''gpu'''
    net.to(device)
    net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    T1 = time.time()
    '''estimate once'''
    # # net.train()
    # w_c2 = net.module.layer1[0].conv1.weight.data
    # out2 = W_Quan_betalaw_new(w_c2, M)
    # net.module.layer1[0].conv1.weight.data = out2[0]

    w_c3 = net.module.layer1[0].conv2.weight.data
    out3 = W_Quan_betalaw_new(w_c3, M)
    net.module.layer1[0].conv2.weight.data = out3[0]
    
    w_c4 = net.module.layer1[1].conv1.weight.data
    out4 = W_Quan_betalaw_new(w_c4, M)
    net.module.layer1[1].conv1.weight.data = out4[0]

    w_c5 = net.module.layer1[1].conv2.weight.data
    out5 = W_Quan_betalaw_new(w_c5, M)
    net.module.layer1[1].conv2.weight.data = out5[0]

    w_c6 = net.module.layer2[0].conv1.weight.data
    out6 = W_Quan_betalaw_new(w_c6, M)
    net.module.layer2[0].conv1.weight.data = out6[0]

    w_c7 = net.module.layer2[0].conv2.weight.data
    out7 = W_Quan_betalaw_new(w_c7, M)
    net.module.layer2[0].conv2.weight.data = out7[0]

    w_c8 = net.module.layer2[1].conv1.weight.data
    out8 = W_Quan_betalaw_new(w_c8, M)
    net.module.layer2[1].conv1.weight.data = out8[0]

    w_c9 = net.module.layer2[1].conv2.weight.data
    out9 = W_Quan_betalaw_new(w_c9, M)
    net.module.layer2[1].conv2.weight.data = out9[0]

    w_c10 = net.module.layer3[0].conv1.weight.data
    out10 = W_Quan_betalaw_new(w_c10, M)
    net.module.layer3[0].conv1.weight.data = out10[0]

    w_c11 = net.module.layer3[0].conv2.weight.data
    out11 = W_Quan_betalaw_new(w_c11, M)
    net.module.layer3[0].conv2.weight.data = out11[0]

    w_c12 = net.module.layer3[1].conv1.weight.data
    out12 = W_Quan_betalaw_new(w_c12, M)
    net.module.layer3[1].conv1.weight.data = out12[0]

    w_c13 = net.module.layer3[1].conv2.weight.data
    out13 = W_Quan_betalaw_new(w_c13, M)
    net.module.layer3[1].conv2.weight.data = out13[0]

    w_c14 = net.module.layer4[0].conv1.weight.data
    out14 = W_Quan_betalaw_new(w_c14, M)
    net.module.layer4[0].conv1.weight.data = out14[0]

    w_c15 = net.module.layer4[0].conv2.weight.data
    out15 = W_Quan_betalaw_new(w_c15, M)
    net.module.layer4[0].conv2.weight.data = out15[0]

    #w_c16 = net.module.layer4[1].conv1.weight.data
    #out16 = W_Quan_betalaw_new(w_c16, M)
    #net.module.layer4[1].conv1.weight.data = out16[0]

    #w_c17 = net.module.layer4[1].conv2.weight.data
    #out17 = W_Quan_betalaw_new(w_c17, M)
    #net.module.layer4[1].conv2.weight.data = out17[0]
    
    net.to(device)
    net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    net = net.module
    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch + 1))
        # net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        loader_train.sampler.set_epoch(epoch)
        loader_test.sampler.set_epoch(epoch)

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
            # print(net)
        # print(net.module.layer1[0].conv1.weight.data)
        #get the ac with testdataset in each epoch
        scheduler.step()
        if epoch != 0:
            # w_c2 = net.module.layer1[0].conv1.weight.data
            # out2 = W_Quan_betalaw_new(w_c2, M, out2[1], out2[2])
            # net.module.layer1[0].conv1.weight.data = out2[0]

            w_c3 = net.module.layer1[0].conv2.weight.data
            out3 = W_Quan_betalaw_new(w_c3, M, out3[1], out3[2])
            net.module.layer1[0].conv2.weight.data = out3[0]
            
            w_c4 = net.module.layer1[1].conv1.weight.data
            out4 = W_Quan_betalaw_new(w_c4, M, out4[1], out4[2])
            net.module.layer1[1].conv1.weight.data = out4[0]

            w_c5 = net.module.layer1[1].conv2.weight.data
            out5 = W_Quan_betalaw_new(w_c5, M, out5[1], out5[2])
            net.module.layer1[1].conv2.weight.data = out5[0]

            w_c6 = net.module.layer2[0].conv1.weight.data
            out6 = W_Quan_betalaw_new(w_c6, M, out6[1], out6[2])
            net.module.layer2[0].conv1.weight.data = out6[0]

            w_c7 = net.module.layer2[0].conv2.weight.data
            out7 = W_Quan_betalaw_new(w_c7, M, out7[1], out7[2])
            net.module.layer2[0].conv2.weight.data = out7[0]

            w_c8 = net.module.layer2[1].conv1.weight.data
            out8 = W_Quan_betalaw_new(w_c8, M, out8[1], out8[2])
            net.module.layer2[1].conv1.weight.data = out8[0]

            w_c9 = net.module.layer2[1].conv2.weight.data
            out9 = W_Quan_betalaw_new(w_c9, M, out9[1], out9[2])
            net.module.layer2[1].conv2.weight.data = out9[0]

            w_c10 = net.module.layer3[0].conv1.weight.data
            out10 = W_Quan_betalaw_new(w_c10, M, out10[1], out10[2])
            net.module.layer3[0].conv1.weight.data = out10[0]

            w_c11 = net.module.layer3[0].conv2.weight.data
            out11 = W_Quan_betalaw_new(w_c11, M, out11[1], out11[2])
            net.module.layer3[0].conv2.weight.data = out11[0]

            w_c12 = net.module.layer3[1].conv1.weight.data
            out12 = W_Quan_betalaw_new(w_c12, M, out12[1], out12[2])
            net.module.layer3[1].conv1.weight.data = out12[0]

            w_c13 = net.module.layer3[1].conv2.weight.data
            out13 = W_Quan_betalaw_new(w_c13, M, out13[1], out13[2])
            net.module.layer3[1].conv2.weight.data = out13[0]

            w_c14 = net.module.layer4[0].conv1.weight.data
            out14 = W_Quan_betalaw_new(w_c14, M, out14[1], out14[2])
            net.module.layer4[0].conv1.weight.data = out14[0]

            w_c15 = net.module.layer4[0].conv2.weight.data
            out15 = W_Quan_betalaw_new(w_c15, M, out15[1], out15[2])
            net.module.layer4[0].conv2.weight.data = out15[0]

            #w_c16 = net.module.layer4[1].conv1.weight.data
            #out16 = W_Quan_betalaw_new(w_c16, M, out16[1], out16[2])
            #net.module.layer4[1].conv1.weight.data = out16[0]

            #w_c17 = net.module.layer4[1].conv2.weight.data
            #out17 = W_Quan_betalaw_new(w_c17, M, out17[1], out17[2])
            #net.module.layer4[1].conv2.weight.data = out17[0]

            net.to(device)
            net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
            net = net.module
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
                print('Test\'s ac is: %.3f%%' % (100 * correct / total))
        

        
    print('Train has finished, total epoch is %d' % epochs)
    T2 = time.time()
    print('time:',(T2-T1)*1000,'ms')
    print('Finished Training')
    # torch.save(net, './VGG16_Quan_multi_conv_2bits.pkl')



if __name__ == "__main__":
    #for M in range(2,6):
    M=4
    print('M=',M)
    run(M)


#     '''分布式训练初始化'''
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#     torch.distributed.init_process_group(backend="nccl")
#     local_rank = torch.distributed.get_rank()
#     torch.cuda.set_device(local_rank)

#     device = torch.device("cuda", local_rank)


#     ##模型处理
#     net = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    

#     '''运行之前记得修改保存文件！！'''
#     epochs = 20
#     batch_size = 256
#     lr = 0.0005
#     M = 2

#     torch.set_grad_enabled(True)
#     net.train()

#     ##数据集处理

#     train_data_path = "/data/dataset/Imagenet2012/train"
#     val_data_path = "/data/dataset/Imagenet2012/val"
#     transform = transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     train_data = datasets.ImageFolder(root=train_data_path, transform=transform)
#     test_data = datasets.ImageFolder(root=val_data_path, transform=transform)

#     sampler_train = DistributedSampler(train_data) 
#     loader_train = DataLoader(train_data, batch_size=4*batch_size, sampler=sampler_train)
#     sampler_test = DistributedSampler(test_data) 
#     loader_test = DataLoader(test_data, batch_size=4*batch_size, sampler=sampler_test)

#     '''gpu'''
#     net.to(device)
#     net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr = lr)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

#     T1 = time.time()
#     '''estimate once'''
#     # # net.train()
#     w_c2 = net.module.layer1[0].conv1.weight.data
#     out2 = W_Quan_betalaw_new(w_c2, M)
#     net.module.layer1[0].conv1.weight.data = out2[0]

#     w_c3 = net.module.layer1[0].conv2.weight.data
#     out3 = W_Quan_betalaw_new(w_c3, M)
#     net.module.layer1[0].conv2.weight.data = out3[0]
    
#     w_c4 = net.module.layer1[1].conv1.weight.data
#     out4 = W_Quan_betalaw_new(w_c4, M)
#     net.module.layer1[1].conv1.weight.data = out4[0]

#     w_c5 = net.module.layer1[1].conv2.weight.data
#     out5 = W_Quan_betalaw_new(w_c5, M)
#     net.module.layer1[1].conv2.weight.data = out5[0]

#     w_c6 = net.module.layer2[0].conv1.weight.data
#     out6 = W_Quan_betalaw_new(w_c6, M)
#     net.module.layer2[0].conv1.weight.data = out6[0]

#     w_c7 = net.module.layer2[0].conv2.weight.data
#     out7 = W_Quan_betalaw_new(w_c7, M)
#     net.module.layer2[0].conv2.weight.data = out7[0]

#     w_c8 = net.module.layer2[1].conv1.weight.data
#     out8 = W_Quan_betalaw_new(w_c8, M)
#     net.module.layer2[1].conv1.weight.data = out8[0]

#     w_c9 = net.module.layer2[1].conv2.weight.data
#     out9 = W_Quan_betalaw_new(w_c9, M)
#     net.module.layer2[1].conv2.weight.data = out9[0]

#     w_c10 = net.module.layer3[0].conv1.weight.data
#     out10 = W_Quan_betalaw_new(w_c10, M)
#     net.module.layer3[0].conv1.weight.data = out10[0]

#     w_c11 = net.module.layer3[0].conv2.weight.data
#     out11 = W_Quan_betalaw_new(w_c11, M)
#     net.module.layer3[0].conv2.weight.data = out11[0]

#     w_c12 = net.module.layer3[1].conv1.weight.data
#     out12 = W_Quan_betalaw_new(w_c12, M)
#     net.module.layer3[1].conv1.weight.data = out12[0]

#     w_c13 = net.module.layer3[1].conv2.weight.data
#     out13 = W_Quan_betalaw_new(w_c13, M)
#     net.module.layer3[1].conv2.weight.data = out13[0]

#     w_c14 = net.module.layer4[0].conv1.weight.data
#     out14 = W_Quan_betalaw_new(w_c14, M)
#     net.module.layer4[0].conv1.weight.data = out14[0]

#     w_c15 = net.module.layer4[0].conv2.weight.data
#     out15 = W_Quan_betalaw_new(w_c15, M)
#     net.module.layer4[0].conv2.weight.data = out15[0]

#     w_c16 = net.module.layer4[1].conv1.weight.data
#     out16 = W_Quan_betalaw_new(w_c16, M)
#     net.module.layer4[1].conv1.weight.data = out16[0]

#     w_c17 = net.module.layer4[1].conv2.weight.data
#     out17 = W_Quan_betalaw_new(w_c17, M)
#     net.module.layer4[1].conv2.weight.data = out17[0]
#     net.to(device)
#     net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
#     net = net.module
#     for epoch in range(epochs):
#         print('\nEpoch: %d' % (epoch + 1))
#         # net.train()
#         sum_loss = 0.0
#         correct = 0.0
#         total = 0.0
#         loader_train.sampler.set_epoch(epoch)
#         loader_test.sampler.set_epoch(epoch)

#         if epoch != 0:
#             w_c2 = net.module.layer1[0].conv1.weight.data
#             out2 = W_Quan_betalaw_new(w_c2, M, out2[1], out2[2])
#             net.module.layer1[0].conv1.weight.data = out2[0]

#             w_c3 = net.module.layer1[0].conv2.weight.data
#             out3 = W_Quan_betalaw_new(w_c3, M, out3[1], out3[2])
#             net.module.layer1[0].conv2.weight.data = out3[0]
            
#             w_c4 = net.module.layer1[1].conv1.weight.data
#             out4 = W_Quan_betalaw_new(w_c4, M, out4[1], out4[2])
#             net.module.layer1[1].conv1.weight.data = out4[0]

#             w_c5 = net.module.layer1[1].conv2.weight.data
#             out5 = W_Quan_betalaw_new(w_c5, M, out5[1], out5[2])
#             net.module.layer1[1].conv2.weight.data = out5[0]

#             w_c6 = net.module.layer2[0].conv1.weight.data
#             out6 = W_Quan_betalaw_new(w_c6, M, out6[1], out6[2])
#             net.module.layer2[0].conv1.weight.data = out6[0]

#             w_c7 = net.module.layer2[0].conv2.weight.data
#             out7 = W_Quan_betalaw_new(w_c7, M, out7[1], out7[2])
#             net.module.layer2[0].conv2.weight.data = out7[0]

#             w_c8 = net.module.layer2[1].conv1.weight.data
#             out8 = W_Quan_betalaw_new(w_c8, M, out8[1], out8[2])
#             net.module.layer2[1].conv1.weight.data = out8[0]

#             w_c9 = net.module.layer2[1].conv2.weight.data
#             out9 = W_Quan_betalaw_new(w_c9, M, out9[1], out9[2])
#             net.module.layer2[1].conv2.weight.data = out9[0]

#             w_c10 = net.module.layer3[0].conv1.weight.data
#             out10 = W_Quan_betalaw_new(w_c10, M, out10[1], out10[2])
#             net.module.layer3[0].conv1.weight.data = out10[0]

#             w_c11 = net.module.layer3[0].conv2.weight.data
#             out11 = W_Quan_betalaw_new(w_c11, M, out11[1], out11[2])
#             net.module.layer3[0].conv2.weight.data = out11[0]

#             w_c12 = net.module.layer3[1].conv1.weight.data
#             out12 = W_Quan_betalaw_new(w_c12, M, out12[1], out12[2])
#             net.module.layer3[1].conv1.weight.data = out12[0]

#             w_c13 = net.module.layer3[1].conv2.weight.data
#             out13 = W_Quan_betalaw_new(w_c13, M, out13[1], out13[2])
#             net.module.layer3[1].conv2.weight.data = out13[0]

#             w_c14 = net.module.layer4[0].conv1.weight.data
#             out14 = W_Quan_betalaw_new(w_c14, M, out14[1], out14[2])
#             net.module.layer4[0].conv1.weight.data = out14[0]

#             w_c15 = net.module.layer4[0].conv2.weight.data
#             out15 = W_Quan_betalaw_new(w_c15, M, out15[1], out15[2])
#             net.module.layer4[0].conv2.weight.data = out15[0]

#             w_c16 = net.module.layer4[1].conv1.weight.data
#             out16 = W_Quan_betalaw_new(w_c16, M, out16[1], out16[2])
#             net.module.layer4[1].conv1.weight.data = out16[0]

#             w_c17 = net.module.layer4[1].conv2.weight.data
#             out17 = W_Quan_betalaw_new(w_c17, M, out17[1], out17[2])
#             net.module.layer4[1].conv2.weight.data = out17[0]

#             net.to(device)
#             net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
#             net = net.module


#         for i, data in enumerate(loader_train, 0):
#             net.train()
#             #prepare dataset
#             length = len(loader_train)
#             # print(length)
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
            
#             #forward & backward
#             outputs = net(inputs)
#             loss = loss_function(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             #print ac & loss in each batch
#             sum_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += predicted.eq(labels.data).cpu().sum()
#             if i % 250 == 249:
#                 print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
#                     % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
#             # print(net)
#         # print(net.module.layer1[0].conv1.weight.data)
#         #get the ac with testdataset in each epoch


#         print('Waiting Test...')
#         with torch.no_grad():
#             correct = 0
#             total = 0
#             for data in loader_test:
#                 net.eval()
#                 images, labels = data
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = net(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum()
#             print('Test\'s ac is: %.3f%%' % (100 * correct / total))

#     print('Train has finished, total epoch is %d' % epochs)
#     T2 = time.time()
#     print('time:',(T2-T1)*1000,'ms')
#     print('Finished Training')
#     # torch.save(net, './VGG16_Quan_multi_conv_2bits.pkl')



