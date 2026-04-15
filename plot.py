import matplotlib.pyplot as plt
import numpy as np
import json
import os
import json
import glob

def load_optimizer_data(directory="."):
    """
    Loads all JSON files from a directory and organizes them by optimizer name.
    Expected file content: {"Dataset1": [loss...], "Dataset2": [loss...]}
    """
    all_results = {}
    
    # Path pattern to find all json files
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    if not json_files:
        print("No JSON files found! Please upload your data to Colab.")
        return None

    for file_path in json_files:
        # Use the filename (without .json) as the Optimizer name
        opt_name = os.path.basename(file_path).replace(".json", "")
        
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                all_results[opt_name] = data
                print(f"Loaded {opt_name} successfully.")
            except json.JSONDecodeError:
                print(f"Error: Could not parse {file_path}")
                
    return all_results


def plot_neurips_comparison(all_results):
    # Get list of datasets and optimizers
    optimizers = list(all_results.keys())
    datasets = list(all_results[optimizers[0]].keys())
    
    # Aesthetics Configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "DejaVu Serif"],
        "axes.labelsize": 13,
        "font.size": 12,
        "legend.fontsize": 10,
        "axes.titlesize": 14,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3
    })

    # Create a figure with one subplot per dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1: axes = [axes] # Ensure axes is iterable
    
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
    markers = ['o', 's', '^', 'D']

    for i, ds in enumerate(datasets):
        ax = axes[i]
        for j, opt in enumerate(optimizers):
            loss_history = all_results[opt][ds]
            epochs = np.arange(1, len(loss_history) + 1)
            
            ax.plot(epochs, loss_history, label=opt, color=colors[j], 
                    marker=markers[j], markersize=4, markevery=2)

        ax.set_title(f"Dataset: {ds}")
        ax.set_xlabel("Epoch")
        if i == 0: ax.set_ylabel("Training Loss")
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle='--')
        ax.legend(frameon=True, loc='upper right')

    plt.tight_layout()
    plt.savefig("optimizer_comparison_neurips.pdf", bbox_inches='tight')
    plt.show()

# Run the plotting function
results_data = load_optimizer_data()
plot_neurips_comparison(results_data)
