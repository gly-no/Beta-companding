import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import json

def beta_pdf_plot(a,b,la = 'PDF'):
    # songti = matplotlib.font_manager.FontProperties(fname = "/home/gly/.conda/envs/quan/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/SIMSUN.TTC",size=15) 

    g = math.gamma(a+b)/(math.gamma(a)*math.gamma(b))
    x = np.arange(0, 1, 0.01)
    pdf = []
    for t in x:
        y_1 = g*np.power(t,a-1)*np.power(1-t,b-1)
        pdf.append(y_1)


    out_hist = dict(zip(x,pdf))
    
    #保存
    with open('latex_plot/vgg_conv1_pdf.json', 'w') as f:
        json.dump(out_hist, f)

    plt.clf()
    plt.plot(x, pdf,label = la, linewidth = 4)

    plt.xlabel("x")
    plt.ylabel("y")
    # plt.ylim(0)
    plt.legend()
    plt.savefig("plot/vgg_conv1_pdf.png")
    # plt.show()

def beta_cdf_plot(a,b, la='cdf'):
    g = math.gamma(a+b)/(math.gamma(a)*math.gamma(b))
    x = np.arange(0.01, 1, 0.01)
    ff = lambda x: math.pow(x, a-1)*math.pow(1-x, b-1)
    cdf = []
    for t in x:
        ww, ev = integrate.quad(ff, 0, t)
        y_1 = g*ww
        cdf.append(y_1)

    out_hist = dict(zip(x,cdf))
    
    #保存
    with open('latex_plot/cdf_a1_b1.json', 'w') as f:
        json.dump(out_hist, f)

    plt.clf()
    plt.plot(x, cdf, label = la)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("plot/cdf_a1_b1.png")
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()



if __name__ == '__main__':
    # a = 0.3389
    # b = 6.9903
    a = 1
    b = 1
    beta_cdf_plot(a,b)
    beta_pdf_plot(a,b)
    c = 0


