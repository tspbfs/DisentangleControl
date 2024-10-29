import matplotlib.pyplot as plt
import numpy as np

def plot_lambda(lam):


    opt_lam = lam[-1] * np.ones(len(lam))
    plt.plot(lam, color='black', label='Online trust parameter'+r' $\lambda_t$', linewidth=3)
    
    plt.legend(loc='best', scatterpoints=1, frameon=True, labelspacing=0.5, prop={'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

def plot_trajectory(y,color):

    plt.plot(y[:,0], y[:,1], linewidth=2, color=color, linestyle='dashed', label='Trajectory'+r' $\mathbf{y}$')
    plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.5, prop={'size': 15}, fancybox=True, framealpha=0.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

def plot_track(x, y,context, color):

    plt.plot(x[:,0]+y[:,0], x[:,1]+y[:,1],  linewidth=3, color=color, label=context)
    plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.5, prop={'size': 15}, fancybox=True, framealpha=0.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)




