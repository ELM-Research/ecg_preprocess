import numpy as np
import matplotlib.pyplot as plt

from utils.file_dir import ensure_directory_exists

def plot_ecg(ecg, leads = None, sf = 250, title = None):
    n_leads, T = ecg.shape
    t = np.arange(T) / sf

    fig, axes = plt.subplots(n_leads, 1, figsize=(12, n_leads * 0.8), sharex = True)
    if n_leads == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, ecg[i], color = 'k', linewidth = 0.5)
        ax.set_ylabel(leads[i], fontsize=8, rotation=0, 
                      ha = "right", va = "center")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    
    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    ensure_directory_exists(folder = "pngs")
    plt.savefig(f"pngs/{title}.png", dpi = 150, bbox_inches = "tight")
    plt.close()