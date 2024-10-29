import matplotlib.pyplot as plt
import numpy as np

def plot_lambda(lam):


    opt_lam = lam[-1] * np.ones(len(lam))
    plt.plot(lam, color='black', label='Online trust parameter'+r' $\lambda_t$', linewidth=3)
    # plt.plot(opt_lam, color='gray', linestyle='dashed', label='Optimal trust parameter' +r' $\lambda^*$', linewidth=2)
    plt.legend(loc='best', scatterpoints=1, frameon=True, labelspacing=0.5, prop={'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
