import numpy as np
import matplotlib.pyplot as plt
def sum_of_sins(n):
    x = np.arange(1, n + 1)
    sin_values = np.sin(x)
    sum_sins = np.cumsum(sin_values)
    result_an = (np.sin((x+1)/2)*np.sin(x/2))/np.sin(1/2)
    return x, sum_sins,result_an

if __name__ == "__main__":
    n = 1*10**12  
    x,result_sum,result_an = sum_of_sins(n)
    plt.plot(x,np.abs(result_sum-result_an))
    plt.yscale('log')
    plt.xlabel('n')
    plt.xscale('log')
    plt.ylabel('resudual')
    plt.show()