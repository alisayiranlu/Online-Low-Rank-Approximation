from final_research_c import *
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict

from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_openml
from collections import defaultdict
import csv
# ---------------------------
# Utility: row normalization
# ---------------------------
def normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    return X / norms

# ---------------------------
# Your MWUA + HRD code here
# ---------------------------
# assume ExpertMWUA, HRD etc. are already defined from your code
# e.g. from your script above, just import or paste those class definitions

# ---------------------------
# Load MNIST
# ---------------------------
def run_MNIST_data():
    print("Downloading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data']
    y = mnist['target'].astype(int)

    print("MNIST shape:", X.shape)  # 70000 by 784

    # ---------------------------
    # Dimensionality reduction
    # ---------------------------
    k = 20 
    d_split = 15
    r_expert = 10
    print(f"Reducing to {k} dimensions with TruncatedSVD...")
    svd = TruncatedSVD(n_components=k, random_state=42)
    X_reduced = svd.fit_transform(X)

    # Normalize to unit vectors
    X_unit = normalize_rows(X_reduced)

    # ---------------------------
    # Stream into ExpertMWUA
    # ---------------------------
    hrd = SphericalHRD(k=k, d_split=d_split, r_expert=r_expert, n_min=20, epsilon_hrd=0.1, n_max_leaf=100)
    mw = ExpertMWUA(hrd, eta=0.5, r_expert=r_expert,
                    candidate_pool_size=12, max_experts=300, combined_basis_dim=r_expert, random_seed=0)
        
    hrd_losses = []
    print("Streaming MNIST vectors into ExpertMWUA...")
    for i, x in enumerate(X_unit[:200]):  # limit to 2000 for speed; adjust as needed
        agg_loss, chosen, basis = mw.step(x)
        hrd_losses.append(agg_loss)
        if (i+1) % 5 == 0:
            print(f"Step {i+1}: AggLoss={agg_loss:.4f}, Basis size={len(chosen)}")
    return {
        'hrd_cumulative': mw.cum_loss[1:],
        'hrd_instantaneous': hrd_losses,
        'num_leaves': len(hrd.leaves)
    }
def run_Credit_Card_data():
    print("Fetching Data...")
    file_path = "creditcard.csv"
    data_list = []
    columns_to_remove = [0]
    k = 20 
    d_split = 15
    r_expert = 2
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        columns_to_remove = []
        for row in reader:
            pruned_row = [row[i] for i in columns_to_remove]
            data_list.append(pruned_row)

    #print("MNIST shape:", X.shape)  # 70000 by 784

    # ---------------------------
    # Stream into ExpertMWUA
    # ---------------------------
    hrd = SphericalHRD(k=k, d_split=d_split, r_expert=r_expert, n_min=20, epsilon_hrd=0.1, n_max_leaf=100)
    mw = ExpertMWUA(hrd, eta=0.5, r_expert=r_expert,
                    candidate_pool_size=12, max_experts=300, combined_basis_dim=r_expert, random_seed=0)
        
    hrd_losses = []
    print("Streaming MNIST vectors into ExpertMWUA...")
    for i, x in enumerate(data_list[:200]):  # limit to 200 for speed; adjust as needed
        agg_loss, chosen, basis = mw.step(x)
        hrd_losses.append(agg_loss)
        if (i+1) % 5 == 0:
            print(f"Step {i+1}: AggLoss={agg_loss:.4f}, Basis size={len(chosen)}")
    return {
        'hrd_cumulative': mw.cum_loss[1:],
        'hrd_instantaneous': hrd_losses,
        'num_leaves': len(hrd.leaves)
    }
def plot_results(results):
    """Create plots comparing the algorithms"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Optimal data experiment
    ax1 = axes[0, 0]
    ax1.plot(results['hrd_cumulative'], label='HRD Algorithm', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def print_summary(results):
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nMNIST Experiment:")
    print(f"  HRD Final Loss: {results['hrd_cumulative'][-1]:.4f}")


def test_performance_benchmark():
    """Run the complete benchmark experiment"""
    print("Starting MNIST Performance Benchmark")
    print("=" * 50)
    
    np.random.seed(42)
    MNIST_results = run_MNIST_data()
    print("DONE with MNIST")
    
    print_summary(MNIST_results)
    
    print("\nGenerating plots...")
    plot_results(MNIST_results)
    print("\nBenchmark completed!")
    return MNIST_results

print("here")
test_performance_benchmark()