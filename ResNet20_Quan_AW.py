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
from solve import W_Quan_betalaw_new,  A_Quan_betalaw_new
import os
import resnet20
from thop import profile
import infobatch

def run(M):
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # device = "cpu"
    ##模型处理
    # net = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

    net = resnet20.ResNet20()
    # print(net)
    # net.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    # print(net)

    net.load_state_dict(torch.load('Resnet20_cifar10.pt'))
    
    # model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    # model_ori = models.resnet18()
    # model_ori.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    # input = torch.randn(256,3,32,32)
    # flops, params = profile(net, inputs=(input, ))
    # print('N2UQ_flops:{}'.format(flops))
    # print('N2UQ_params:{}'.format(params))


    # # input = torch.randn(256,3,32,32)
    # flops, params = profile(model_ori, inputs=(input, ))
    # print('ori_flops:{}'.format(flops))
    # print('ori_params:{}'.format(params))


    '''gpu'''
    net.to(device)

    '''运行之前记得修改保存文件！！'''
    epoch = 100
    batch_size = 256
    lr = 0.005
    # M = 1

    torch.set_grad_enabled(True)
    net.train()

    ##数据集
  
    # transform = transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    transform_train = transforms.Compose([
                                    # transforms.RandomRotation(),  # 随机旋转
                                    transforms.RandomCrop(32, padding=4),  # 填充后裁剪
                                    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    # transforms.ColorJitter(brightness=1),  # 颜色变化。亮度
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose([
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),#Q1数据归一化问题：ToTensor是把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])#Q2均值方差问题：RGB3个通道分别进行正则化处理

    train_data= datasets.CIFAR10(root='/data/dataset',train=True,transform=transform_train,download=True)
    
    '''change here'''
    train_data = infobatch.InfoBatch(train_data, epoch, 0.2)
    
    train_loader = Data.DataLoader(train_data,batch_size = batch_size,shuffle=False,num_workers=2,drop_last=True,sampler=train_data.sampler)
    test_data = torchvision.datasets.CIFAR10(root='/data/dataset',train=False,transform=transform_test,download=True)
    test_loader = Data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=2,drop_last=True)
    

    
    loss_function = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr = lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    T1 = time.time()
    w_c2 = net.layer1[0].left[0].weight.data
    out2 = W_Quan_betalaw_new(w_c2, M)
    net.layer1[0].left[0].weight.data = out2[0]

    w_c3 = net.layer1[0].left[3].weight.data
    out3 = W_Quan_betalaw_new(w_c3, M)
    net.layer1[0].left[3].weight.data = out3[0]
    
    w_c4 = net.layer1[1].left[0].weight.data
    out4 = W_Quan_betalaw_new(w_c4, M)
    net.layer1[1].left[0].weight.data = out4[0]

    w_c5 = net.layer1[1].left[3].weight.data
    out5 = W_Quan_betalaw_new(w_c5, M)
    net.layer1[1].left[3].weight.data = out5[0]

    w_c6 = net.layer1[2].left[0].weight.data
    out6 = W_Quan_betalaw_new(w_c6, M)
    net.layer1[2].left[0].weight.data = out6[0]

    w_c7 = net.layer1[2].left[3].weight.data
    out7 = W_Quan_betalaw_new(w_c7, M)
    net.layer1[2].left[3].weight.data = out7[0]

    w_c8 = net.layer2[0].left[0].weight.data
    out8 = W_Quan_betalaw_new(w_c8, M)
    net.layer2[0].left[0].weight.data = out8[0]

    w_c9 = net.layer2[0].left[3].weight.data
    out9 = W_Quan_betalaw_new(w_c9, M)
    net.layer2[0].left[3].weight.data = out9[0]

    w_c10 = net.layer2[1].left[0].weight.data
    out10 = W_Quan_betalaw_new(w_c10, M)
    net.layer2[1].left[0].weight.data = out10[0]

    w_c11 = net.layer2[1].left[3].weight.data
    out11 = W_Quan_betalaw_new(w_c11, M)
    net.layer2[1].left[3].weight.data = out11[0]

    w_c12 = net.layer2[2].left[0].weight.data
    out12 = W_Quan_betalaw_new(w_c12, M)
    net.layer2[2].left[0].weight.data = out12[0]

    w_c13 = net.layer2[2].left[3].weight.data
    out13 = W_Quan_betalaw_new(w_c13, M)
    net.layer2[2].left[3].weight.data = out13[0]

    w_c14 = net.layer3[0].left[0].weight.data
    out14 = W_Quan_betalaw_new(w_c14, M)
    net.layer3[0].left[0].weight.data = out14[0]

    w_c15 = net.layer3[0].left[3].weight.data
    out15 = W_Quan_betalaw_new(w_c15, M)
    net.layer3[0].left[3].weight.data = out15[0]

    w_c16 = net.layer3[1].left[0].weight.data
    out16 = W_Quan_betalaw_new(w_c16, M)
    net.layer3[1].left[0].weight.data = out16[0]

    w_c17 = net.layer3[1].left[3].weight.data
    out17 = W_Quan_betalaw_new(w_c17, M)
    net.layer3[1].left[3].weight.data = out17[0]

    w_c18 = net.layer3[2].left[0].weight.data
    out18 = W_Quan_betalaw_new(w_c18, M)
    net.layer3[2].left[0].weight.data = out18[0]


    def act_hook(module,input,output):
        # print(f"{module} input: {input}")
        # print(f"{module} output: {output}")
        out = A_Quan_betalaw_new(output, M)
        output = out[0].to(device)
        # return output.to(device)
    
    hook1 = net.layer1[0].left[2].register_forward_hook(act_hook)
    hook2 = net.layer1[1].left[2].register_forward_hook(act_hook)
    hook3 = net.layer1[2].left[2].register_forward_hook(act_hook)
    hook4 = net.layer2[0].left[2].register_forward_hook(act_hook)
    hook5 = net.layer2[1].left[2].register_forward_hook(act_hook)
    hook6 = net.layer2[2].left[2].register_forward_hook(act_hook)
    hook7 = net.layer3[0].left[2].register_forward_hook(act_hook)
    hook8 = net.layer3[1].left[2].register_forward_hook(act_hook)
    hook9 = net.layer3[2].left[2].register_forward_hook(act_hook)

    for epoch in range(epoch):
        running_loss = 0
        running_corrects = 0
        for step, data in enumerate(train_loader):
            torch.set_grad_enabled(True)#在接下来的计算中每一次运算产生的节点都是可以求导的
            net.train()
    
            net.to(device)
            optimizer.zero_grad()
            input, label = data
            # indices, rescale_weight = indices.to(device), rescale_weight.to(device)

            output = net(input.to(device))            
            loss = loss_function(output, label.to(device))

            '''change here'''
            # train_data.__setscore__(indices.detach().cpu().numpy(),loss.detach().cpu().numpy())
            # loss = loss*rescale_weight
            # loss = torch.mean(loss)
            loss = train_data.update(loss)

            
            loss.backward()
            running_loss += loss.item()
            pred = output.data.argmax(dim=1)
            running_corrects += torch.sum(pred.cpu() == label.data)
            optimizer.step()
            if step % 50 == 49:
                print('Epoch', epoch+1,',step', step+1,'|Loss_avg:',running_loss/(step+1),'|Acc_avg:',running_corrects/((step+1)*batch_size))
                '''test acc'''
            hook1.remove()
            hook2.remove()
            hook3.remove()
            hook4.remove()
            hook5.remove()
            hook6.remove()
            hook7.remove()
            hook8.remove()
            hook9.remove()
        
        if epoch != 0:
            hook1 = net.layer1[0].left[2].register_forward_hook(act_hook)
            hook2 = net.layer1[1].left[2].register_forward_hook(act_hook)
            hook3 = net.layer1[2].left[2].register_forward_hook(act_hook)
            hook4 = net.layer2[0].left[2].register_forward_hook(act_hook)
            hook5 = net.layer2[1].left[2].register_forward_hook(act_hook)
            hook6 = net.layer2[2].left[2].register_forward_hook(act_hook)
            hook7 = net.layer3[0].left[2].register_forward_hook(act_hook)
            hook8 = net.layer3[1].left[2].register_forward_hook(act_hook)
            hook9 = net.layer3[2].left[2].register_forward_hook(act_hook)            

            w_c2 = net.layer1[0].left[0].weight.data
            out2 = W_Quan_betalaw_new(w_c2, M, out2[1],out2[2])
            # out2.to(device)
            net.layer1[0].left[0].weight.data = out2[0]
            net.layer1[0].left[0].weight.grad = net.layer1[0].left[0].weight.grad.cpu() * out2[3]

            w_c3 = net.layer1[0].left[3].weight.data
            out3 = W_Quan_betalaw_new(w_c3, M, out3[1],out3[2])
            net.layer1[0].left[3].weight.data = out3[0]
            net.layer1[0].left[3].weight.grad = net.layer1[0].left[3].weight.grad.cpu() * out3[3]
            
            w_c4 = net.layer1[1].left[0].weight.data
            out4 = W_Quan_betalaw_new(w_c4, M, out4[1],out4[2])
            net.layer1[1].left[0].weight.data = out4[0]
            net.layer1[1].left[0].weight.grad = net.layer1[1].left[0].weight.grad.cpu() * out4[3]

            w_c5 = net.layer1[1].left[3].weight.data
            out5 = W_Quan_betalaw_new(w_c5, M, out5[1],out5[2])
            net.layer1[1].left[3].weight.data = out5[0]
            net.layer1[1].left[3].weight.grad = net.layer1[1].left[3].weight.grad.cpu() * out5[3]

            w_c6 = net.layer1[2].left[0].weight.data
            out6 = W_Quan_betalaw_new(w_c6, M, out6[1],out6[2])
            net.layer1[2].left[0].weight.data = out6[0]
            net.layer1[2].left[0].weight.grad = net.layer1[2].left[0].weight.grad.cpu() * out6[3]

            w_c7 = net.layer1[2].left[3].weight.data
            out7 = W_Quan_betalaw_new(w_c7, M, out7[1],out7[2])
            net.layer1[2].left[3].weight.data = out7[0]
            net.layer1[2].left[3].weight.grad = net.layer1[2].left[3].weight.grad.cpu() * out7[3]

            w_c8 = net.layer2[0].left[0].weight.data
            out8 = W_Quan_betalaw_new(w_c8, M, out8[1],out8[2])
            net.layer2[0].left[0].weight.data = out8[0]
            net.layer2[0].left[0].weight.grad = net.layer2[0].left[0].weight.grad.cpu() * out8[3]

            w_c9 = net.layer2[0].left[3].weight.data
            out9 = W_Quan_betalaw_new(w_c9, M, out9[1],out9[2])
            net.layer2[0].left[3].weight.data = out9[0]
            net.layer2[0].left[3].weight.grad = net.layer2[0].left[3].weight.grad.cpu() * out9[3]

            w_c10 = net.layer2[1].left[0].weight.data
            out10 = W_Quan_betalaw_new(w_c10, M, out10[1],out10[2])
            net.layer2[1].left[0].weight.data = out10[0]
            net.layer2[1].left[0].weight.grad = net.layer2[1].left[0].weight.grad.cpu() * out10[3]

            w_c11 = net.layer2[1].left[3].weight.data
            out11 = W_Quan_betalaw_new(w_c11, M, out11[1],out11[2])
            net.layer2[1].left[3].weight.data = out11[0]
            net.layer2[1].left[3].weight.grad = net.layer2[1].left[3].weight.grad.cpu() * out11[3]

            w_c12 = net.layer2[2].left[0].weight.data
            out12 = W_Quan_betalaw_new(w_c12, M, out12[1],out12[2])
            net.layer2[2].left[0].weight.data = out12[0]
            net.layer2[2].left[0].weight.grad = net.layer2[2].left[0].weight.grad.cpu() * out12[3]

            w_c13 = net.layer2[2].left[3].weight.data
            out13 = W_Quan_betalaw_new(w_c13, M, out13[1],out13[2])
            net.layer2[2].left[3].weight.data = out13[0]
            net.layer2[2].left[3].weight.grad = net.layer2[2].left[3].weight.grad.cpu() * out13[3]

            w_c14 = net.layer3[0].left[0].weight.data
            out14 = W_Quan_betalaw_new(w_c14, M, out14[1],out14[2])
            net.layer3[0].left[0].weight.data = out14[0]
            net.layer3[0].left[0].weight.grad = net.layer3[0].left[0].weight.grad.cpu() * out14[3]

            w_c15 = net.layer3[0].left[3].weight.data
            out15 = W_Quan_betalaw_new(w_c15, M, out15[1],out15[2])
            net.layer3[0].left[3].weight.data = out15[0]
            net.layer3[0].left[3].weight.grad = net.layer3[0].left[3].weight.grad.cpu() * out15[3]

            w_c16 = net.layer3[1].left[0].weight.data
            out16 = W_Quan_betalaw_new(w_c16, M, out16[1],out16[2])
            net.layer3[1].left[0].weight.data = out16[0]
            net.layer3[1].left[0].weight.grad = net.layer3[1].left[0].weight.grad.cpu() * out16[3]

            w_c17 = net.layer3[1].left[3].weight.data
            out17 = W_Quan_betalaw_new(w_c17, M, out17[1],out17[2])
            net.layer3[1].left[3].weight.data = out17[0]
            net.layer3[1].left[3].weight.grad = net.layer3[1].left[3].weight.grad.cpu() * out17[3]

            w_c18 = net.layer3[2].left[0].weight.data
            out18 = W_Quan_betalaw_new(w_c18, M, out18[1],out18[2])
            net.layer3[2].left[0].weight.data = out18[0]
            net.layer3[2].left[0].weight.grad = net.layer3[2].left[0].weight.grad.cpu() * out18[3]



        net.to(device)
        torch.set_grad_enabled(False)
        net.eval()
        ##更新
        acc = 0.0
        for i, data in enumerate(test_loader):
            x, y = data
            y_pred = net(x.to(device, torch.float))
            pred_t = y_pred.argmax(dim=1)
            acc += (pred_t.data == y.data.to(device)).sum()
            if i==0:
                hook1.remove()
                hook2.remove()
                hook3.remove()
                hook4.remove()
                hook5.remove()
                hook6.remove()
                hook7.remove()
                hook8.remove()
                hook9.remove()
        acc = (acc / 10000) * 100
        print('Accuracy: %.2f' %acc, '%')

        # running_loss = 0
        # scheduler.step()
    T2 = time.time()
    # print('time:',(T2-T1)*1000,'ms')
    print('Finished Training')
    # torch.save(net.state_dict(), 'Resnet18_cifar10.pt')



if __name__ == "__main__":
    for M in range(2,6):
        print(M)
        run(M)
        # run(4)

