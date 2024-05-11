import os

import matplotlib.pyplot as plt
import pandas as pd

# Set the path to the csv with the IS and FID results for each architecture and dataset
mnist_ddpm_csv = "./modules/diffusion_models/evaluation/csv/mnist.csv"
mnist_bfn_csv = "./modules/bayesian_flow_networks/evaluation/csv/mnist.csv"
cifar10_ddpm_csv = "./modules/diffusion_models/evaluation/csv/cifar10.csv"
cifar10_bfn_csv = "./modules/bayesian_flow_networks/evaluation/csv/cifar10.csv"

# Set the path to the directory where the plots will be saved
save_path = "./assets/plots/"


def plot_results(ddpm_results, bfn_results, score, dataset):
    plt.figure()
    ddpm_score = ddpm_results[" "+score]
    bfn_score = bfn_results[" "+score]
    epochs_ddpm = ddpm_results["epoch"]
    epochs_bfn = bfn_results["epoch"]
    plt.plot(epochs_ddpm, ddpm_score, label="DDPM")
    plt.plot(epochs_bfn, bfn_score, label="BFN")
    plt.xlabel("Epochs")
    plt.ylabel(f"{score} (on {dataset})")
    plt.legend()
    plt.savefig(fname=os.path.join(
        save_path, f"{score}_{dataset}_training_curves.png"))


if __name__ == "__main__":
    ddpm_mnist_results = pd.read_csv(mnist_ddpm_csv)
    bfn_mnist_results = pd.read_csv(mnist_bfn_csv)
    ddpm_cifar10_results = pd.read_csv(cifar10_ddpm_csv)
    bfn_cifar10_results = pd.read_csv(cifar10_bfn_csv)

    # Publication-ready pyplot theme
    plot_settings = {'ytick.labelsize': 16,
                     'xtick.labelsize': 16,
                     'font.size': 22,
                     'figure.figsize': (10, 5),
                     'axes.titlesize': 22,
                     'axes.labelsize': 18,
                     'lines.linewidth': 2,
                     'lines.markersize': 3,
                     'legend.fontsize': 11,
                     'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral'}
    # apply defined settings to current pyplot style
    plt.style.use(plot_settings)

    os.makedirs(save_path, exist_ok=True)
    for score in "IS", "FID":
        plot_results(ddpm_mnist_results, bfn_mnist_results, score, "MNIST")
        plot_results(ddpm_cifar10_results,
                     bfn_cifar10_results, score, "CIFAR10")
    print(f"Done, plots saved at {save_path}")
