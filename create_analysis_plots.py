"""
Mira Welner
August 2025

This script loads the heartrate data generated in generate_processed_heartrate.py and displays the histograms and the heartrates themselves.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#data processing params
patient_names = '06-31-24', '09-40-14', '10-48-45', '11-03-38', '13-22-23', '14-17-50'
split_locs = [-0.75, -0.555, -0.16], -0.5, -0.1, None ,-0.5,-0.37
snip_len = 2500

def get_histograms(n_bins, name='dist_hist'):
    """
    Load data based on patient names and display histograms of the
    heartrate distributions. Display the locations in which the distributions are split using
    vertical lines.
    """
    _, axes = plt.subplots(len(patient_names)//2,2, figsize=(15,6), layout = "constrained")
    axes = axes.flatten()
    gmm = GaussianMixture(n_components=n_bins, random_state=42)
    for itr, (patient,split) in enumerate(zip(patient_names,split_locs)):
        heartrate_data = np.loadtxt(f'processed_data/heartrate_{patient}_unscaled.csv', delimiter=',')

        # Fit GMM to raw data
        gmm.fit(heartrate_data.reshape(-1, 1))

        # Make histogram
        hist, bin_edges = np.histogram(heartrate_data, bins=200)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Predict label for each bin center
        bin_labels = gmm.predict(bin_centers.reshape(-1, 1))

        last_indices = np.sort(np.array([np.max(np.where(bin_labels == val)) for val in np.unique(bin_labels)]))[:-1]
        # Plot
        axes[itr].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
        axes[itr].set_title(f"Patient {patient}")
        axes[itr].set_xlim(0.5, 1.25)
        axes[itr].vlines(bin_edges[:-1][last_indices],ymin=0, ymax=max(hist), color='red')
        if itr==0:
            axes[itr].set_xlabel("R-R interval length (s)")
            axes[itr].set_ylabel("Frequency in recording")
    plt.suptitle("R-R Interval Histograms for 6 Patients split by Gaussian Mixture")
    plt.savefig("figures/dist_hist.png")
    plt.show()

def get_dist_plots(name='dist_plot'):
    if len(patient_names)//2:
        _, axes = plt.subplots(len(patient_names)//2,2, figsize=(15,6), layout = "constrained")
        axes = axes.flatten()
    else:
        _, axes = plt.subplots(1,1, figsize=(15,6), layout = "constrained")
        axes = [axes]
    for itr, (patient,split) in enumerate(zip(patient_names,split_locs)):
        heartrate_data = np.loadtxt(f'processed_data/heartrate_{patient}_unscaled.csv',delimiter=',')
        heartrate_data = heartrate_data[:-2] #remove min and max
        axes[itr].plot(heartrate_data)
        axes[itr].set_title(f"Patient {patient} - distribution split at {split}")
        #if split:
        #    axes[itr].hlines(split,xmin=0, xmax=len(heartrate_data), color='red')

        axes[itr].set_xticks(range(0,len(heartrate_data), 2*10*60*60), [str(i) for i in range(0,int(np.ceil(len(heartrate_data)/(10*60*60))), 2)])
        if itr==0:
            axes[itr].set_ylabel("scaled r-r interval length")
            axes[itr].set_xlabel("Time (h)")
    plt.savefig(f"figures/{name}.png")
    plt.show()

get_histograms(2)
get_dist_plots()
