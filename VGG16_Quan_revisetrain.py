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
from solve import W_Quan_betalaw_new
import os


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # device = "cpu"
    ##模型处理
    net = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
    # print(net)
    net.classifier[0] = nn.Linear(in_features=25088, out_features=512, bias=True)
    net.classifier[3] = nn.Linear(in_features=512, out_features=512, bias=True)
    net.classifier[6] = nn.Linear(in_features=512, out_features=100, bias=True)
    # torch.nn.init.normal_(net.classifier[6].weight.data, mean=0.0, std=1.0)
    # print(net)
    # print(net.classifier)

    # w_c1 = net.features[0].weight.data.reshape(1,-1)[0].numpy()
    # w_c2 = net.features[2].weight.data.reshape(1,-1)[0].numpy()
    # w_c3 = net.features[5].weight.data.reshape(1,-1)[0].numpy()
    # w_c4 = net.features[7].weight.data.reshape(1,-1)[0].numpy()
    # w_c5 = net.features[10].weight.data.reshape(1,-1)[0].numpy()

    # plt.rcParams['font.sans-serif'] = ['SIMSUN']
    # plt.clf()
    
    # plt.hist(w_c1, bins= 100, alpha = 1, label= '第一层',density= True)
    # plt.hist(w_c2, bins= 100, alpha = 1, label= '第二层',density= True)
    # plt.hist(w_c3, bins= 100, alpha = 1, label= '第三层',density= True)
    # plt.hist(w_c4, bins= 100, alpha = 1, label= '第四层',density= True)
    # plt.hist(w_c5, bins= 100, alpha = 1, label= '第五层',density= True)
 
    # plt.xticks(fontsize=12,weight='bold')
    # plt.yticks(fontsize=12,weight='bold')  
    # plt.rcParams.update({'font.size': 12})
    # plt.legend(loc = 'upper right', prop = {'weight':"bold"})
    # plt.show()
    # plt.savefig("plot/conv_weight_distri.svg")
    '''gpu'''
    net.to(device)
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # net = net.cuda()
    # net = torch.nn.parallel.DistributedDataParallel(net)
    # net = torch.nn.DataParallel(net)

    w_c2 = net.features[0].weight.data
    out2 = W_Quan_betalaw_new(w_c2, 4)

    '''运行之前记得修改保存文件！！'''
    epoch = 60
    batch_size = 128
    lr = 0.0001
    M = 4

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


    T1 = time.time()
    for epoch in range(epoch):
        running_loss = 0
        running_corrects = 0

        '''fully-connection layers'''
        # w_f1 = net.classifier[0].weight.data
        # _,_,a1,b1 = W_Quan_betalaw_new(w_f1, M, train_or_not = True)

        '''convolution layers'''
        # w_c2 = net.features[2].weight.data
        # _,_,a12,b12 = W_Quan_betalaw_new(w_c2, M, train_or_not = True)

        # w_c3 = net.features[5].weight.data
        # _,_,a13,b13 = W_Quan_betalaw_new(w_c3, M, train_or_not = True)

        # w_c4 = net.features[7].weight.data
        # _,_,a14,b14 = W_Quan_betalaw_new(w_c4, M, train_or_not = True)

        # w_c5 = net.features[10].weight.data
        # _,_,a15,b15 = W_Quan_betalaw_new(w_c5, M, train_or_not = True)

        # w_c6 = net.features[12].weight.data
        # _,_,a16,b16 = W_Quan_betalaw_new(w_c6, M, train_or_not = True)

        # w_c7 = net.features[14].weight.data
        # _,_,a17,b17 = W_Quan_betalaw_new(w_c7, M, train_or_not = True)
        for step, data in enumerate(train_loader):
            torch.set_grad_enabled(True)#在接下来的计算中每一次运算产生的节点都是可以求导的
            net.train()
            '''fc----quan'''
            # w_f1 = net.classifier[0].weight.data
            # out_f1 = W_Quan_betalaw_new(w_f1, M)
            # net.classifier[0].weight.data = out_f1[0]
            # w_f2 = net.classifier[3].weight.data
            # out_f2 = W_Quan_betalaw_new(w_f2, M)
            # net.classifier[3].weight.data = out_f2[0]

            '''conv----quan----beta_law'''
            # w_c2 = net.features[2].weight.data
            # out2 = W_Quan_betalaw_new(w_c2, M)
            # # out2 = W_Quan_Kumaraswamy(w_c2, M)
            # net.features[2].weight.data = out2[0]

            # w_c3 = net.features[5].weight.data
            # out3 = W_Quan_betalaw_new(w_c3, M)
            # # out3 = W_Quan_Kumaraswamy(w_c3, M)
            # net.features[5].weight.data = out3[0]
            
            # w_c4 = net.features[7].weight.data
            # out4 = W_Quan_betalaw_new(w_c4, M)
            # net.features[7].weight.data = out4[0]

            # w_c5 = net.features[10].weight.data
            # out5 = W_Quan_betalaw_new(w_c5, M)
            # net.features[10].weight.data = out5[0]

            # w_c6 = net.features[12].weight.data
            # out6 = W_Quan_betalaw_new(w_c6, M)
            # net.features[12].weight.data = out6[0]

            # w_c7 = net.features[14].weight.data
            # out7 = W_Quan_betalaw_new(w_c7, M)
            # net.features[14].weight.data = out7[0]

            # w_c8 = net.features[17].weight.data
            # out8 = W_Quan_betalaw_new(w_c8, M)
            # net.features[17].weight.data = out8[0]

            # w_c9 = net.features[19].weight.data
            # out9 = W_Quan_betalaw_new(w_c9, 8)
            # net.features[19].weight.data = out9[0]

            # w_c10 = net.features[21].weight.data
            # out10 = W_Quan_betalaw_new(w_c10, 8)
            # net.features[21].weight.data = out10[0]

            # w_c11 = net.features[24].weight.data
            # out11 = W_Quan_betalaw_new(w_c11, 8)
            # net.features[24].weight.data = out11[0]

            # w_c12 = net.features[26].weight.data
            # out12 = W_Quan_betalaw_new(w_c12, 8)
            # net.features[26].weight.data = out12[0]

            # w_c13 = net.features[28].weight.data
            # out13 = W_Quan_betalaw_new(w_c13, 8)
            # net.features[28].weight.data = out13[0]

            '''conv----quan----normal'''
            # w_c2 = net.features[2].weight.data
            # out2 = W_Quan_normal(w_c2, M, train_or_not = True)
            # net.features[2].weight.data = out2[0]

            # w_c3 = net.features[5].weight.data
            # out3 = W_Quan_normal(w_c3, M, train_or_not = True)
            # net.features[5].weight.data = out3[0]
            
            # w_c4 = net.features[7].weight.data
            # out4 = W_Quan_normal(w_c4, M, train_or_not = True)
            # net.features[7].weight.data = out4[0]

            # w_c5 = net.features[10].weight.data
            # out5 = W_Quan_normal(w_c5, M, train_or_not = True)
            # net.features[10].weight.data = out5[0]

            # w_c6 = net.features[12].weight.data
            # out6 = W_Quan_normal(w_c6, M, train_or_not = True)
            # net.features[12].weight.data = out6[0]

            # w_c7 = net.features[14].weight.data
            # out7 = W_Quan_normal(w_c7, M, train_or_not = True)
            # net.features[14].weight.data = out7[0]

            # w_c8 = net.features[17].weight.data
            # out8 = W_Quan_normal(w_c8, M, train_or_not = True)
            # net.features[17].weight.data = out8[0]


            # w_c14 = net.features[28].weight.data
            # out14 = W_Quan_betalaw_new(w_c14, M, train_or_not = True)
            # net.features[28].weight.data = out14[0]

            net.to(device)
            input, label = data
            optimizer.zero_grad()

            output = net(input.to(device))            
            loss = loss_function(output, label.to(device))
            loss.backward()

            '''grad'''
            # gg = net.features[2].weight.grad
            # out_g2 = torch.from_numpy(np.float32(out2[1]))
            # net.features[2].weight.grad = out_g2.to(device)*gg

            # gg = net.features[5].weight.grad
            # out_g3 = torch.from_numpy(np.float32(out3[1]))
            # net.features[5].weight.grad = out_g3.to(device)*gg

            # gg = net.features[7].weight.grad
            # out_g4 = torch.from_numpy(np.float32(out4[1]))
            # net.features[7].weight.grad = out_g4.to(device)*gg

            # gg = net.features[10].weight.grad
            # out_g5 = torch.from_numpy(np.float32(out5[1]))
            # net.features[10].weight.grad = out_g5.to(device)*gg

            # gg = net.features[12].weight.grad
            # out_g6 = torch.from_numpy(np.float32(out6[1]))
            # net.features[12].weight.grad = out_g6.to(device)*gg

            # gg = net.features[14].weight.grad
            # out_g7 = torch.from_numpy(np.float32(out7[1]))
            # net.features[14].weight.grad = out_g7.to(device)*gg

            # gg = net.features[17].weight.grad
            # out_g8 = torch.from_numpy(np.float32(out8[1]))
            # net.features[17].weight.grad = out_g8.to(device)*gg

            '''grad normal'''
            # gg = net.features[2].weight.grad
            # out_g2 = out2[1].to(torch.float32)
            # net.features[2].weight.grad = out_g2.to(device)*gg

            # gg = net.features[5].weight.grad
            # out_g3 = torch.np.float32(out3[1])
            # net.features[5].weight.grad = out_g3.to(device)*gg

            # gg = net.features[7].weight.grad
            # out_g4 = torch.np.float32(out4[1])
            # net.features[7].weight.grad = out_g4.to(device)*gg

            # gg = net.features[10].weight.grad
            # out_g5 = torch.np.float32(out5[1])
            # net.features[10].weight.grad = out_g5.to(device)*gg

            # gg = net.features[12].weight.grad
            # out_g6 = torch.np.float32(out6[1])
            # net.features[12].weight.grad = out_g6.to(device)*gg

            # gg = net.features[14].weight.grad
            # out_g7 = torch.np.float32(out7[1])
            # net.features[14].weight.grad = out_g7.to(device)*gg

            # gg = net.features[17].weight.grad
            # out_g8 = torch.np.float32(out8[1])
            # net.features[17].weight.grad = out_g8.to(device)*gg

            # gg = net.features[28].weight.grad
            # out_g14 = torch.from_numpy(np.float32(out14[1]))
            # net.features[28].weight.grad = out_g14.to(device)*gg
            
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
    T2 = time.time()
    print('time:',(T2-T1)*1000,'ms')
    print('Finished Training')
    # torch.save(net, './VGG16_Quan_multi_conv_2bits.pkl')
