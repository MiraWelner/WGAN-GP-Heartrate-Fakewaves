"""
Mira Welner
August 2025

This script loads the heartrate data generated in generate_processed_heartrate.py and displays the histograms and the heartrates themselves.
It comes with many other functions to display the binning study, elbow graphs, silhouette study, etc.

The patient names and 'elbows' are hardcoded at the top.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from fastkde.fastKDE import pdf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f

#data processing params
patient_names = '06-31-24', '09-40-14', '10-48-45', '11-03-38', '13-22-23', '14-17-50'
elbows = 3, 2, 2, 2, 2, 2


def variance_ratio_gmm_1d(n, n_components):
    X = n.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(X)
    means = gmm.means_.reshape(-1)           # shape (K,)
    resp = gmm.predict_proba(X)              # shape (num_samples, K)
    overall_mean = X.mean()
    tss = np.sum((X.flatten() - overall_mean) ** 2)
    diffs = X - means.reshape(1, -1)
    sq_dists = diffs ** 2
    wss = np.sum(resp * sq_dists)
    bss = tss - wss
    variance_ratio = bss / tss
    return variance_ratio


def f_test(name='variance_ratio', max_splits=10):
    _, axes = plt.subplots(2, len(patient_names)//2, figsize=(14,7), layout = "constrained")
    axes = axes.flatten()
    n_components_range = range(1, max_splits)
    for itr, (patient,elbow) in enumerate(zip(patient_names,elbows)):
        # Load your 1D heartrate data
        heartrate_data = np.loadtxt(f'processed_data/heartrate_{patient}_unscaled.csv', delimiter=',')
        all_ratios = []
        for ncomp in n_components_range:
            all_ratios.append(variance_ratio_gmm_1d(heartrate_data, ncomp))
        axes[itr].plot(n_components_range, all_ratios)
        axes[itr].set_ylabel('Variance Ratio')
        axes[itr].set_xlabel('# Gaussian Distributions in Mixture')
        axes[itr].set_title(f"patient {patient}")
    plt.suptitle('Variance Ratio - Goodness of Fit')
    plt.savefig(f"figures/{name}.png")
    plt.show()





def binning_study(name='binning', bin_nums = [10,50,100,400,700,1000,2000]):
    _, axes = plt.subplots(len(patient_names), len(bin_nums), figsize=(17,10), layout = "constrained")
    for itr, (patient,elbow) in enumerate(zip(patient_names,elbows)):
        # Load your 1D heartrate data
        heartrate_data = np.loadtxt(
            f'processed_data/heartrate_{patient}_unscaled.csv',
            delimiter=','
        )
        for j, bin_num in enumerate(bin_nums):
            if j == 0:
                axes[itr,j].set_ylabel(patient)
            if itr == 0:
                axes[itr,j].set_title(f"{bin_num} bins")
            axes[itr,j].hist(heartrate_data, bins=bin_num)
            axes[itr,j].set_xticks([])
            axes[itr,j].set_yticks([])

    plt.suptitle("Binning Study")
    plt.savefig(f"figures/{name}.png")
    plt.show()

def silhouette_plot(name='silhouette', samplings=50):
    _ = plt.figure(figsize=(14,7), layout = "constrained")
    for itr, (patient,elbow) in enumerate(zip(patient_names,elbows)):
        heartrate_data = np.loadtxt(f'processed_data/heartrate_{patient}_unscaled.csv', delimiter=',').reshape(-1, 1)

        silhouettes = []
        for _ in range(samplings):
            indices = np.random.choice(len(heartrate_data), len(heartrate_data)//50, replace=False)
            sampled_data = heartrate_data[indices]
            kmeans = KMeans(n_clusters=elbow)
            cluster_labels = kmeans.fit_predict(sampled_data)
            avg_silhouette = silhouette_score(sampled_data, cluster_labels)
            silhouettes.append(avg_silhouette)
        plt.scatter(range(len(silhouettes)), silhouettes, label=f"{patient}: {elbow} splits")
    plt.ylabel("silhouette score")
    plt.xlabel("nth sample")
    plt.legend()
    plt.title("Silhouette Scores of Subsampled Patient Data (each sample 1/50th of all data)")
    plt.savefig(f"figures/{name}.png")
    plt.show()


def violin_plot(name='violin'):
    all_data = []
    for patient in patient_names:
        heartrate_data = np.loadtxt(
            f'processed_data/heartrate_{patient}_unscaled.csv',
            delimiter=','
        ).ravel()
        all_data.append(heartrate_data)


    plt.figure(figsize=(14,7), layout = "constrained")
    _ = plt.violinplot(all_data, showmeans=True, showextrema=True, showmedians=True)

    plt.xticks(range(1, len(patient_names) + 1), patient_names)
    plt.xlabel("Patient")
    plt.ylabel("Heartrate")
    plt.title("Heartrate Distributions Violin Plot")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.savefig(f"figures/{name}.png")
    plt.show()


def kernel_estimate(name='kernel'):
    """
    Uses fastKDE to create and plot a Gaussian Kernel Density Estimate
    """
    _, axes = plt.subplots(2, len(patient_names)//2, figsize=(14,7), layout = "constrained")
    axes = axes.flatten()
    for itr, patient in enumerate(patient_names):
        # Load your 1D heartrate data
        heartrate_data = np.loadtxt(
            f'processed_data/heartrate_{patient}_unscaled.csv',
            delimiter=','
        ).ravel()

        density = np.array(pdf(heartrate_data))
        axes[itr].set_yticks([])
        axes[itr].plot(density)
        axes[itr].set_xlabel('Heartrate')
        axes[itr].set_ylabel('Estimated Density')
        axes[itr].set_title(f"Patient {patient}")
    plt.suptitle("Gaussian Kernel Density Function for 6 patients")
    plt.savefig(f"figures/{name}.png")
    plt.show()

def get_elbow_graph(name='elbow', max_splits = 10):
    _, axes = plt.subplots(2, len(patient_names)//2, figsize=(14,7), layout = "constrained")
    axes = axes.flatten()
    for itr, (patient,elbow) in enumerate(zip(patient_names,elbows)):
        heartrate_data = np.loadtxt(f'processed_data/heartrate_{patient}_unscaled.csv', delimiter=',').reshape(-1, 1)
        n_components_range = range(1, max_splits)
        bics = []
        aics = []
        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(heartrate_data)
            bics.append(gmm.bic(heartrate_data))
            aics.append(gmm.aic(heartrate_data))

        axes[itr].set_yticks([])
        axes[itr].plot(n_components_range, bics, marker='o', label='Bayesian Information Criterion (BIC)')
        axes[itr].plot(n_components_range, aics, marker='s', label='Akaike Information Criterion (AIC)')
        axes[itr].set_xlabel('Number of Components')
        axes[itr].set_ylabel('Score')
        axes[itr].vlines([elbow], ymin=min(bics), ymax=max(bics), color='black')
        axes[itr].set_title(f"Patient {patient}")
        axes[itr].legend()
    plt.suptitle("Elbow Plots for Gaussian Mixture Models of 6 Patients Heartrates")
    plt.savefig(f"figures/{name}.png")
    plt.show()


def histograms(name='histogram_plot'):
    """
    Load data based on patient names and display histograms of the
    heartrate distributions. Display the locations in which the distributions are split using
    vertical lines.
    """
    _, axes = plt.subplots(2, len(patient_names)//2, figsize=(14,7), layout = "constrained")
    axes = axes.flatten()
    for itr, (patient,elbow) in enumerate(zip(patient_names,elbows)):
        heartrate_data = np.loadtxt(f'processed_data/heartrate_{patient}_unscaled.csv', delimiter=',')

        # Fit GMM to raw data
        gmm = GaussianMixture(n_components=elbow)
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
    plt.savefig(f"figures/{name}.png")
    plt.show()

def plot_heartrate_over_time(name='dist_plot'):
    if len(patient_names)//2:
        _, axes = plt.subplots(len(patient_names)//2,2, figsize=(15,6), layout = "constrained")
        axes = axes.flatten()
    else:
        _, axes = plt.subplots(1,1, figsize=(15,6), layout = "constrained")
        axes = [axes]
    for itr, (patient,split) in enumerate(zip(patient_names,elbows)):
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

histograms()
