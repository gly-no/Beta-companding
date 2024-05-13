import torch
import torch.nn as nn
from scipy.optimize import minimize, root
import numpy as np
import math
from model import LeNet
from model import FCnet
from numpy import *
from sympy import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from solve_test import train as solution
from solve_test import sci_solve, suf_st_beta, suf_st_normal, suf_st_kumaraswamy
import scipy.stats as st
from scipy import integrate
import pdf_cdf_plot as pcplot
import json


def A_Quan_betalaw_new(w, M, a1 = None, b1 = None, a=None, name=None):
    # # w_n = W_normalization(w)
    w1 = w.cpu().detach().numpy()
    # # b = np.ceil(np.abs(w1).max())
    r = w1.max()
    l = w1.min()
    # ww1 = (w1/b + 1)/2 
    # lb = np.exp(-5)
    # rb = np.exp(-0.01)

    w_s = torch.sign(w)
    w_p  = torch.abs(w)
    # w_ppp = torch.abs(w)
    w_pp = w_p.flatten()
    w_pp, index_p = w_pp.ravel()[torch.nonzero(w_pp)][:,0].sort()

    if a == None:
        a = w_pp[round(len(w_pp) * 0.9)]
    else:
        a = a

    w_p = torch.minimum(w_p, (w * 0 + 1 ) * a) / a 
    # w_p = (w_p * w_s + 1)/2

    w_pp = w_p.flatten()
    w_pp = w_pp.ravel()[torch.nonzero(w_pp)][:,0]

    #(1-a)%直接量化为1，用a%做映射
    temp = w_pp - 1
    w_pp = temp.flatten()
    w_pp = w_pp.ravel()[torch.nonzero(w_pp)][:,0] + 1
    w_pp = w_pp.cpu().detach().numpy()

    if a1 == None:
        a11, b11 = suf_st_beta(w_pp)
        a11 = a11.numpy()
        b11 = b11.numpy()
    else:
        a11 = a1
        b11 = b1


    # pcplot.beta_pdf_plot(a11,b11)
    '''here'''
    w_p = w_p.cpu().detach().numpy()
    w_s = w_s.cpu().detach().numpy()
    a = a.cpu().detach().numpy()
    ####mapping
    # cdf1 = st.beta.cdf(w_pp, a11, b11)
    cdf1 = st.beta.cdf(w_p, a11, b11)
    pdf1 = st.beta.pdf(w_p, a11, b11)
    
    ###uniform quantization
    w_q = np.round((math.pow(2,M)-1)*cdf1)/(math.pow(2,M)-1)
   
    # ww1_Q = w_q * a

    '''here'''
    '''creat look-up table'''
    tab = np.power(2, M) - 1
    dic = {}
    for i in range(tab + 1):
        dic[i/tab] = st.beta.isf(1-i/tab, a11, b11)
          
    ###inverse mapping
    # ss1 = 1 - w_q
    # ww1_Q_1 = st.beta.isf(ss1, a11, b11)
    # ww1_Q_1 = ww1_Q_1.clip(ww1.min(),ww1.max())
    shape = w_q.shape
    ww1_Q_1 = np.zeros(shape)
    for idx, data in np.ndenumerate(w_q):
        ww1_Q_1[idx] = dic[data]
    # ww1_Q_1 = ww1_Q_1.clip(ww1.min(),ww1.max())
    # ww1_Q_1 = ww1_Q_1.clip(lb, rb)
    ww1_Q = ww1_Q_1 * a
    # ww1_Q = ww1_Q.clip(-1, 1)
    '''here'''  

    # hh = w1.reshape(-1,1)
    
    # h = hh[:-1:100]

    hist, edge = np.histogram(ww1_Q_1.reshape(1,-1)[0],bins=100, density=True)
    av_edge = [0]*100
    for i in range(100):
        av_edge[i] = (edge[i+1] + edge[i]) / 2
    out_hist = dict(zip(av_edge,hist))
    
    #保存
    with open('latex_plot/resnet20_quan_r1_hist.json', 'w') as f:
        json.dump(out_hist, f)


    # # plt.rcParams['font.sans-serif'] = ['SIMSUN']
    # plt.clf()
    # # pcplot.beta_pdf_plot(a11,b11)
    # # plt.hist(w_pp.reshape(1,-1)[0], bins= 100, alpha = 1, label= 'A_1 with 0,1',density= True)  
    # # plt.hist(cdf1.reshape(1,-1)[0], bins= 100, alpha = 1, label= 'mapped activations',density= True)
    # plt.hist(ww1_Q_1.reshape(1,-1)[0], bins= 100, alpha = 1, label= 'quantized activations',density= True) 
    # # plt.hist(ww1_Q_1.reshape(1,-1)[0], bins= 100, alpha = 1, label= 'quantized activations',density= True) 
    # plt.xticks(fontsize=12,weight='bold')
    # plt.yticks(fontsize=12,weight='bold')  
    # plt.rcParams.update({'font.size': 12})
    # plt.legend(loc = 'upper right', prop = {'weight':"bold"})
    # plt.show()
    # # plt.savefig("plot/activation_dis_01.svg")
    # plt.savefig("plot/resnet20quan_r1_hist_no01.png")
    # # plt.savefig("plot/activation_dis_01.eps")
    

    return torch.from_numpy(ww1_Q).to(torch.float32), a11, b11, torch.from_numpy(pdf1).to(torch.float32), w_pp


def W_Quan_betalaw_new(w, M, a1 = None, b1 = None):
    # w_n = W_normalization(w)
    w1 = w.cpu().numpy()
    # b = np.ceil(np.abs(w1).max())
    b = np.abs(w1).max()
    ww1 = (w1/b + 1)/2 
    lb = np.exp(-5)
    rb = np.exp(-0.01)
    ####mapping
    if a1 == None:
        a11, b11 = suf_st_beta(ww1)
        a11 = a11.numpy()
        b11 = b11.numpy()
    else:
        a11 = a1
        b11 = b1

    # pcplot.beta_pdf_plot(a11,b11)

    cdf1 = st.beta.cdf(ww1, a11, b11)
    pdf1 = st.beta.pdf(ww1, a11, b11)
    
    ###uniform quantization
    w_q = np.round((math.pow(2,M)-1)*cdf1)/(math.pow(2,M)-1)
    # ma = cdf1.max()
    # mi = cdf1.min()
    # w_q = (np.ceil(math.pow(2,M)*cdf1.clip(0.0000001, 1))*2-1)/math.pow(2,M+1)
    # w_qq = w_q*(ma - mi) + mi
    ww1_Q = (w_q * 2 - 1) * b

    '''here'''
    '''creat look-up table'''
    tab = np.power(2, M) - 1
    dic = {}
    for i in range(tab + 1):
        dic[i/tab] = st.beta.isf(1-i/tab, a11, b11)
            
    ###inverse mapping
    # ss1 = 1 - w_q
    # ww1_Q_1 = st.beta.isf(ss1, a11, b11)
    # ww1_Q_1 = ww1_Q_1.clip(ww1.min(),ww1.max())
    shape = w_q.shape
    ww1_Q_1 = np.zeros(shape)
    for idx, data in np.ndenumerate(w_q):
        ww1_Q_1[idx] = dic[data]
    ww1_Q_1 = ww1_Q_1.clip(ww1.min(),ww1.max())
    # ww1_Q_1 = ww1_Q_1.clip(lb, rb)
    ww1_Q = (ww1_Q_1 * 2 - 1) * b
    # ww1_Q = ww1_Q.clip(-1, 1)


    hist, edge = np.histogram(ww1_Q.reshape(1,-1)[0], bins=100, density=True)
    av_edge = [0]*100
    for i in range(100):
        av_edge[i] = (edge[i+1] + edge[i]) / 2
    out_hist = dict(zip(av_edge,hist))
    
    #保存
    with open('latex_plot/vgg16_conv1_hist_quan.json', 'w') as f:
        json.dump(out_hist, f)


    # # plt.clf()
    # plt.hist(ww1.reshape(1,-1)[0], bins= 100, alpha = 1, label= 'orignal',density= True)
    # # plt.hist(cdf1.reshape(1,-1)[0], bins= 100, alpha = 1, label= 'normalzation',density= True) 
    # # plt.hist(ww1.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'normalization',density= True) 
    # # plt.hist(cdf1.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'uniform',density= True)
    # # plt.hist(w_q.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'uni_quan',density= True) 
    # # plt.hist(ww1_Q_1.reshape(1,-1)[0], bins= 100, alpha = 0.5, label= 'quantization',density= True) 
    # plt.legend(loc = 'upper right')
    # plt.savefig("plot/vgg16_conv1_hist.png")
    # # plt.show()
    
    # # g1 = np.power(ww1/ww1_Q_1, a11-1)
    # # g2 = np.power((1-ww1)/(1-ww1_Q_1), b11-1)
    # # grad1 = g1*g2
    # # grad1 = grad1.clip(-1000000,1000000)
    return torch.from_numpy(ww1_Q).to(torch.float32), a11, b11, torch.from_numpy(pdf1).to(torch.float32)


def W_normalization(w):
    e = 0.00001
    mean = torch.mean(w)
    var = torch.var(w)
    w = (w - mean)/(torch.sqrt(var)+e)

    return w

def W_learn_clip_beta_n2u(w, M, train_or_not = True, a1p = None, b1p = None, a1n = None, b1n = None, a = None):
    w_s = torch.sign(w)
    w_p = torch.abs(w)
    w_pp = w_p.flatten()
    w_pp, index_p = w_pp.ravel()[torch.nonzero(w_pp)][:,0].sort()

    if a == None:
        a = w_pp[round(len(w_pp) * 0.95)]
    else:
        a = a

    w_p = torch.minimum(w_p, (w * 0 + 1 ) * a) / a 
    w_p = (w_p * w_s + 1)/2

    #(1-a)%直接量化为1，用a%做映射
    temp = w_p - 1
    w_pp = temp.flatten()
    w_pp = w_pp.ravel()[torch.nonzero(w_pp)][:,0] + 1
    w_pp = w_pp.cpu().numpy()

    if train_or_not == True:
        a11p, b11p = suf_st_beta(w_pp)
        a11p = a11p.numpy()
        b11p = b11p.numpy()
    else:
        a11p = a1p
        b11p = b1p


    w_p_Q = nonuniform_quan(w_p=w_p, w_pp=w_pp, a=a, M=M, a11 = a11p, b11 = b11p) 
    w_Q = w_s.cpu() * w_p_Q

    return w_Q, a11p, b11p, a

def nonuniform_quan(w_p, w_pp, a, M, a11 = None, b11 = None):
    w1 = w_p.cpu().numpy()

    cdf1 = st.beta.cdf(w1, a11, b11)
    
    ###uniform quantization
    w_q = np.round((math.pow(2,M)-1)*cdf1)/(math.pow(2,M)-1)

    '''creat look-up table'''
    tab = np.power(2, M) - 1
    dic = {}
    for i in range(tab + 1):
        dic[i/tab] = st.beta.isf(1-i/tab, a11, b11)

    ###inverse mapping
    shape = w_q.shape
    ww1_Q_1 = np.zeros(shape)
    for idx, data in np.ndenumerate(w_q):
        ww1_Q_1[idx] = dic[data] * a
    # ww1_Q_1 = ww1_Q_1.clip(ww1.min(),ww1.max())
    # ww1_Q = (ww1_Q_1 * 2 - 1)*b / 10
    # ww1_Q = ww1_Q.clip(-1, 1)

    return torch.from_numpy(ww1_Q_1).to(torch.float32)


def grad_a(a, w):
    w_s = torch.sign(w)
    w_p = abs(w)
    w_p = w_p - a
    w_p = torch.minimum(w_p, w * 0)
    w_pp = torch.sign(w_s * w_p)
    return w_pp










if __name__ == "__main__":
    # w1 = np.array([[0.2, 0.3, 0.8],[0.6,0.5,0.1]])
    # solve_para(w=w1)
    model = FCnet()
    out = W_Quan_betalaw(model=model, M=2)
    # gg = W_grad(model=out[0],grad=out[1:3])
    print(out[0])
# model.F1.weight.data = w1
