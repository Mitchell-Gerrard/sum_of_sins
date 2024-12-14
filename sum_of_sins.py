import numpy as np
import matplotlib.pyplot as plt
def sum_of_sins(n):
    x = np.arange(1, n + 1)
    sin_values = np.sin(x)
    sum_sins = np.cumsum(sin_values)
    result_an = (np.sin((x+1)/2)*np.sin(x/2))/np.sin(1/2)
    return x, sum_sins,result_an
def sum_of_sins_samples(n, sample_rate):
    # Calculate the logarithmic sampling rate
    x = np.arange(1, n + 1)
    log_indices = np.unique(np.logspace(0, np.log10(n-1), num=int(np.log10(n)*100), dtype=int))
    x_sampled = x[log_indices]
    x=0
    sum_sins = np.zeros_like( x_sampled, dtype=float)
    cumulative_sum = 0
    sin_values = np.sin(np.arange(1, n + 1))  # Precompute sin values

    for i, val in enumerate(x_sampled):
        print(i,len(x_sampled))
        cumulative_sum = np.sum(sin_values[:val])  # Use slicing to calculate partial sums efficiently
        sum_sins[i] = cumulative_sum
    # Memory-efficient cumulative sum computation with sampling
    '''
    sum_sins = []
    cumulative_sum = 0
    current_index = 0

    for i in range(1, n+1 ):
        print(i)
        cumulative_sum += np.sin(i)
        if current_index<len(x):
            if i == x[current_index]:
                sum_sins.append(cumulative_sum)
                current_index += 1
        else:
            break  
    '''
    result_an = (np.sin((x_sampled + 1) / 2) * np.sin(x_sampled / 2)) / np.sin(1 / 2)

    return x_sampled, sum_sins, result_an

    # Convert the list to a NumPy array
    sum_sins = np.array(sum_sins)

    # Analytical result computation for sampled x
    result_an = (np.sin((x + 1) / 2) * np.sin(x / 2)) / np.sin(1 / 2)

    return x_sampled, sum_sins, result_an
if __name__ == "__main__":
    power=9
    n = 10**power
    results_reds = []
    #x = np.memmap('x.dat', dtype='float64', mode='w+', shape=(n,))
    #result_sum = np.memmap('result_sum.dat', dtype='float64', mode='w+', shape=(n,))
    #result_an = np.memmap('result_an.dat', dtype='float64', mode='w+', shape=(n,))

    #x[:], result_sum[:], result_an[:] = sum_of_sins(n)
    #mean_residual = result_sum - result_an
    #x,result_sum,result_an = sum_of_sins(n)
    #chunk_size = 10**3  # Adjust chunk size based on available memory
    num_samples = 1000  # Number of samples to plot
    x_sampled, result_sum_sampled, result_an_sampled = sum_of_sins_samples(n, num_samples)
    print(x_sampled, result_sum_sampled, result_an_sampled)
    #x_sampled, result_sum_sampled, result_an_sampled = sum_of_sins_chunked(n, chunk_size, num_samples)
    mean_residual_sampled = result_sum_sampled - result_an_sampled



    # Calculate statistics

    #coeffs = np.polyfit(x, mean_residual, 10)
    #linear_fit = np.polyval(coeffs, x)
    '''
    log_indices = np.unique(np.logspace(0, np.log10(n-1), num=1000, dtype=int))
    x_sampled = x[log_indices]
    mean_residual_sampled = mean_residual[log_indices]
    '''
    for i in range(power):
        #plt.plot(x, mean_residual, label='Mean Residual')
        plt.plot(x_sampled, mean_residual_sampled, label='Mean Residual')
        #plt.plot(x, linear_fit, label='Linear Fit', linestyle='--')
        plt.xlabel('n')
        plt.xscale('log')
        plt.yscale('symlog', linthresh=1e-16)
        plt.xlim(10**i, 10**(i+1))
        plt.ylabel('Residual')

        plt.savefig(f'sum_of_sins_{i}.png')
        plt.clf()
