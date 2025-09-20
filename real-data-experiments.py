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


def run_MNIST_data():
    print("Downloading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data']
    y = mnist['target'].astype(int)

    print("MNIST shape:", X.shape)  # 70000 by 784


    k = 20 
    d_split = 15
    r_expert = 10
    print(f"Reducing to {k} dimensions with TruncatedSVD...")
    svd = TruncatedSVD(n_components=k, random_state=42)
    X_reduced = svd.fit_transform(X)

    X_unit = normalize_rows(X_reduced)


    hrd = SphericalHRD(k=k, d_split=d_split, r_expert=r_expert, n_min=20, epsilon_hrd=0.1, n_max_leaf=100)
    mw = ExpertMWUA(hrd, eta=0.5, r_expert=r_expert,
                    candidate_pool_size=12, max_experts=300, combined_basis_dim=r_expert, random_seed=0)
    
    badnet = BadNetBaseline(k=k, r=r_expert) #this is the bad net generated in experiments class 
    
 
    hrd_losses = []
    badnet_losses = []
    
    print("Streaming MNIST vectors into both algorithms...")
    for i, x in enumerate(X_unit[:200]):  # limit to 200 for speed, adjust as needed
        agg_loss, chosen, basis = mw.step(x)
        hrd_losses.append(agg_loss)
        
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
        
    badnet = BadNetBaseline(k=k, r=r_expert)
    
    hrd_losses = []
    badnet_losses = []
    
    print("Streaming Credit Card Data vectors into both algorithms...")
    for i, x in enumerate(X_unit[:200]):
        agg_loss, chosen, basis = mw.step(x)
        hrd_losses.append(agg_loss)
        
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
    """Generate two separate plot images"""
    
    # MNIST Plot - separate figure
    plt.figure(figsize=(10, 6))
    plt.plot(mnist_results['hrd_cumulative'], label='HRD Algorithm', 
             linewidth=2, color='blue')
    plt.plot(mnist_results['badnet_cumulative'], label='BadNet Baseline', 
             linewidth=2, color='red', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Loss')
    plt.title('MNIST Dataset - Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mnist_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Credit Card Plot - separate figure
    plt.figure(figsize=(10, 6))
    plt.plot(credit_card_results['hrd_cumulative'], label='HRD Algorithm', 
             linewidth=2, color='blue')
    plt.plot(credit_card_results['badnet_cumulative'], label='BadNet Baseline', 
             linewidth=2, color='red', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Loss')
    plt.title('Credit Card Dataset - Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('creditcard_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Generated two separate image files:")
    print("  - mnist_performance.png")
    print("  - creditcard_performance.png")

def print_summary(results, dataset_name):
    print("\n" + "="*60)
    print(f"{dataset_name} EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"  HRD Final Loss: {results['hrd_cumulative'][-1]:.4f}")
    print(f"  BadNet Final Loss: {results['badnet_cumulative'][-1]:.4f}")
    print(f"  Performance Gap: {results['hrd_cumulative'][-1] - results['badnet_cumulative'][-1]:.4f}")
    print(f"  Loss Ratio (HRD/BadNet): {results['hrd_cumulative'][-1] / results['badnet_cumulative'][-1]:.3f}")
    
    if results['hrd_cumulative'][-1] < results['badnet_cumulative'][-1]:
        print("  ✓ HRD outperforms BadNet baseline")
    else:
        print("  ✗ BadNet baseline outperforms HRD")
    
    print(f"  Number of HRD Leaves: {results['num_leaves']}")
    
def print_summary(results, dataset_name):
    print("\n" + "="*60)
    print(f"{dataset_name} EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"  HRD Final Loss: {results['hrd_cumulative'][-1]:.4f}")
    print(f"  BadNet Final Loss: {results['badnet_cumulative'][-1]:.4f}")
    print(f"  Performance Gap: {results['hrd_cumulative'][-1] - results['badnet_cumulative'][-1]:.4f}")
    print(f"  Loss Ratio (HRD/BadNet): {results['hrd_cumulative'][-1] / results['badnet_cumulative'][-1]:.3f}")
    
    if results['hrd_cumulative'][-1] < results['badnet_cumulative'][-1]:
        print("  ✓ HRD outperforms BadNet baseline")
    else:
        print("  ✗ BadNet baseline outperforms HRD")
    
    print(f"  Number of HRD Leaves: {results['num_leaves']}")

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
    
    # Side-by-side plots
    print("\nGenerating side-by-side plots...")
    plot_results(MNIST_results, credit_card_results)
    
    print("\nBenchmark completed!")
    return MNIST_results, credit_card_results


test_performance_benchmark()