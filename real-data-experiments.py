from final_research_c import *
from experiments import *
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
    
    d_split = 15
    r_expert = 2
    
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        
        print("Header:", header)
        print("Total columns:", len(header))
        
        columns_to_remove = [0, len(header)-2, len(header)-1]  # [0, 29, 30] are not necessary cols 
        
        for row in reader:
            pruned_row = [float(row[i]) for i in range(len(row)) if i not in columns_to_remove]
            data_list.append(pruned_row)
    
    X = np.array(data_list)
    k = X.shape[1]  


    
    X_unit = normalize_rows(X)
    
    hrd = SphericalHRD(k=k, d_split=d_split, r_expert=r_expert, n_min=20, epsilon_hrd=0.1, n_max_leaf=100)
    mw = ExpertMWUA(hrd, eta=0.5, r_expert=r_expert,
                    candidate_pool_size=12, max_experts=300, combined_basis_dim=r_expert, random_seed=0)
        
    # BadNet Baseline
    badnet = BadNetBaseline(k=k, r=r_expert)
    
    # ---------------------------
    # Stream data through both algorithms
    # ---------------------------
    hrd_losses = []
    badnet_losses = []
    
    print("Streaming Credit Card Data vectors into both algorithms...")
    for i, x in enumerate(X_unit[:200]):
        # HRD Algorithm
        agg_loss, chosen, basis = mw.step(x)
        hrd_losses.append(agg_loss)
        
        # BadNet Baseline
        badnet_loss = badnet.step(x)
        badnet_losses.append(badnet_loss)
        
        if (i+1) % 25 == 0:
            print(f"Step {i+1}: HRD Loss={agg_loss:.4f}, BadNet Loss={badnet_loss:.4f}, Basis size={len(chosen)}")
    
    return {
        'hrd_cumulative': mw.cum_loss[1:],
        'hrd_instantaneous': hrd_losses,
        'badnet_cumulative': badnet.cumulative_loss[1:],
        'badnet_instantaneous': badnet_losses,
        'num_leaves': len(hrd.leaves)
    }
def plot_results(mnist_results, credit_card_results):
    """Create single plot comparing cumulative losses for both datasets"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # MNIST Results
    ax.plot(mnist_results['hrd_cumulative'], label='MNIST - HRD Algorithm', 
            linewidth=2, color='blue')
    
    # Credit Card Results  
    ax.plot(credit_card_results['hrd_cumulative'], label='Credit Card - HRD Algorithm', 
            linewidth=2, color='green')
    ax.plot(credit_card_results['badnet_cumulative'], label='Credit Card - BadNet Baseline', 
            linewidth=2, color='red', linestyle='--')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Loss')
    ax.set_title('Cumulative Loss Comparison Across Datasets')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_summary(results, dataset_name):
    print("\n" + "="*60)
    print(f"{dataset_name} EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"  HRD Final Loss: {results['hrd_cumulative'][-1]:.4f}")
    
    # Only print BadNet results if they exist
    if 'badnet_cumulative' in results:
        print(f"  BadNet Final Loss: {results['badnet_cumulative'][-1]:.4f}")
        print(f"  Performance Gap: {results['hrd_cumulative'][-1] - results['badnet_cumulative'][-1]:.4f}")
        print(f"  Loss Ratio (HRD/BadNet): {results['hrd_cumulative'][-1] / results['badnet_cumulative'][-1]:.3f}")
        
        if results['hrd_cumulative'][-1] < results['badnet_cumulative'][-1]:
            print("  ✓ HRD outperforms BadNet baseline")
        else:
            print("  ✗ BadNet baseline outperforms HRD")

def test_performance_benchmark():
    """Run the complete benchmark experiment"""
    print("Starting Performance Benchmark")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Run MNIST
    print("Running MNIST dataset...")
    MNIST_results = run_MNIST_data()
    print("DONE with MNIST")
    print_summary(MNIST_results, "MNIST")

    # Run Credit Card
    print("\nRunning Credit Card dataset...")
    credit_card_results = run_Credit_Card_data()
    print("DONE with Credit Card")
    print_summary(credit_card_results, "CREDIT CARD")
    
    # Single plot with all results
    print("\nGenerating combined plot...")
    plot_results(MNIST_results, credit_card_results)
    
    print("\nBenchmark completed!")
    return MNIST_results, credit_card_results


test_performance_benchmark()