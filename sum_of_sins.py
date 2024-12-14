import numpy as np
import matplotlib.pyplot as plt
def sum_of_sins(n):
    x = np.arange(1, n + 1)
    sin_values = np.sin(x)
    sum_sins = np.cumsum(sin_values)
    result_an = (np.sin((x+1)/2)*np.sin(x/2))/np.sin(1/2)
    return x, sum_sins,result_an

if __name__ == "__main__":
    n = 10**8
    results_reds = []

    x,result_sum,result_an = sum_of_sins(n)
    mean_residual = result_sum - result_an


    # Calculate statistics


    plt.plot(x, mean_residual, label='Mean Residual')


    plt.xlabel('n')
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e-16)
    plt.ylabel('Residual')
    plt.legend()
    plt.show()
