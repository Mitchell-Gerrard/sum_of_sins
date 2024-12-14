import numpy as np
import matplotlib.pyplot as plt
def sum_of_sins(n):
    x = np.arange(1, n + 1)
    sin_values = np.sin(x)
    sum_sins = np.cumsum(sin_values)
    result_an = (np.sin((x+1)/2)*np.sin(x/2))/np.sin(1/2)
    return x, sum_sins,result_an

if __name__ == "__main__":
    power=8
    n = 10**power
    results_reds = []
    #x = np.memmap('x.dat', dtype='float64', mode='w+', shape=(n,))
    #result_sum = np.memmap('result_sum.dat', dtype='float64', mode='w+', shape=(n,))
    #result_an = np.memmap('result_an.dat', dtype='float64', mode='w+', shape=(n,))

    x[:], result_sum[:], result_an[:] = sum_of_sins(n)
    mean_residual = result_sum - result_an
    x,result_sum,result_an = sum_of_sins(n)
    mean_residual = result_sum - result_an


    # Calculate statistics

    #coeffs = np.polyfit(x, mean_residual, 10)
    #linear_fit = np.polyval(coeffs, x)
    for i in range(power):
        plt.plot(x, mean_residual, label='Mean Residual')

        #plt.plot(x, linear_fit, label='Linear Fit', linestyle='--')
        plt.xlabel('n')
        plt.xscale('log')
        plt.yscale('symlog', linthresh=1e-16)
        plt.xlim(10**i, 10**(i+1))
        plt.ylabel('Residual')
        plt.legend()
        plt.savefig(f'sum_of_sins_{power}.png')
        plt.clf()
