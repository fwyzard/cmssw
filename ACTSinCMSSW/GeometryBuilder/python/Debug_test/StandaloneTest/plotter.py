#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_histogram_with_gaussian(file_input, title, n_bins=30):
    """
    Reads a txt file with numbers, plots a histogram, fits a Gaussian,
    and prints/shows mean and standard deviation.
    
    Parameters:
        file_input (str): Path to the input txt file
        n_bins (int): Number of bins for the histogram
    """
    # --- READ FILE ---
    # Assume one number per line
    data = np.loadtxt(file_input)

    # --- HISTOGRAM ---
    counts, bins, patches = plt.hist(data, bins=n_bins, density=True,
                                     alpha=0.6, color='skyblue', label='Data')

    # --- GAUSSIAN FIT ---
    mu, std = norm.fit(data)  # mean and standard deviation
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2, label=f'Gaussian Fit\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')

    # --- LABELS AND LEGEND ---
    plt.xlabel(title)
    # plt.yscale('log')
    plt.xlim(-10,10)
    plt.title(title)
    plt.legend()

    # --- PRINT MEAN AND STD ---
    print(f"Values for {title}:")
    print(f"Mean: {mu:.4f}")
    print(f"Standard Deviation: {std:.4f}")

    # --- SHOW PLOT ---
    plt.savefig(f"{title}_plot.png")


if __name__ == "__main__":
    
    titles = ["phi_res", "phi_pull", "qOverP_res", "qOverP_pull", "theta_res", "theta_pull"]
    # titles = ["qOverP_pull"]
    n_bins = 30
    
    for title in titles:
        this_file = title + '.txt'
        plt.figure()
        plot_histogram_with_gaussian(this_file, title, n_bins)

