from final_research6 import *
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

class BadNetBaseline:
    """Fixed bad net that doesn't adapt over time. Each coordinate is a multiple of +- 1/3, then normalized"""
    def __init__(self, k, r):
        self.k = k
        self.r = r
        self.basis = []

        for i in range(r):
            coords = []
            for j in range(k):
                mult = (j + 1) / 3.0
                sign = 1 if np.random.rand() > 0.5 else -1
                coords.append(sign * mult)
            
            bad_vector = normalize(np.array(coords))
            self.basis.append(bad_vector)

        self.cumulative_loss = [0.0]
    
    def step(self, x): #loss accumulates on the same vector 
        loss = projection_loss(self.basis, x)
        self.cumulative_loss.append(self.cumulative_loss[-1] + loss)
        return loss

def generate_optimal_data_sequence(T, k=5):
    """
    Each vector has only first 2 coordinates non-zero.
    Optimal LRA basis should be [(1,0,0,0,0), (0,1,0,0,0)]
    """
    sequence = []
    for t in range(T):
        x1 = np.random.randn()
        x2 = np.random.randn()
        x = np.array([x1, x2, 0.0, 0.0, 0.0])
        
        x_normalized = normalize(x)
        sequence.append(x_normalized)
    
    return sequence

def run_optimal_data_experiment():
    k = 5  
    d_split = 4 #is less than k
    r_expert = 2  
    T = 1000  
    
    data_sequence = generate_optimal_data_sequence(T, k)
    
    hrd = SphericalHRD(k=k, d_split=d_split, r_expert=r_expert, n_min=20, epsilon_hrd=0.1, n_max_leaf=100)
    mw_hrd = ExpertMWUA(hrd, eta=0.5, r_expert=r_expert,
                    candidate_pool_size=12, max_experts=300, combined_basis_dim=r_expert, random_seed=0)
    
    bad_baseline = BadNetBaseline(k, r_expert)
    
    hrd_losses = []
    bad_losses = []
    
    for t, x in enumerate(data_sequence):
        hrd_loss = mw_hrd.step(x)[0]
        hrd_losses.append(hrd_loss)
        
        bad_loss = bad_baseline.step(x)
        bad_losses.append(bad_loss)
        if (t % 100 == 0):
            print(t)

    
    return {
        'hrd_cumulative': mw_hrd.cum_loss[1:],
        'bad_cumulative': bad_baseline.cumulative_loss[1:],
        'hrd_instantaneous': hrd_losses,
        'bad_instantaneous': bad_losses,
        'num_leaves': len(hrd.leaves)
    }

def run_comparison_experiment():
    """Run comparison between HRD on optimal data and on other data"""
    
    k = 5
    d_split = 3
    r_expert = 2
    T = 1000
    
    results = {}
    
    #clustered
    clustered_data = []
    centers = [normalize(np.random.randn(k)) for _ in range(3)]
    for t in range(T): #make 1000 points around these 3 centers
        center = centers[t % 3]
        noise = 0.3 * np.random.randn(k)
        x = normalize(center + noise)
        clustered_data.append(x)
    
    results['clustered'] = compare_algorithms(clustered_data, k, d_split, r_expert)
    
    #random uniform data
    random_data = [normalize(np.random.randn(k)) for _ in range(T)]
    results['random'] = compare_algorithms(random_data, k, d_split, r_expert)
    
    return results

def compare_algorithms(data_sequence, k, d_split, r_expert):
    
    hrd = SphericalHRD(k=k, d_split=d_split, r_expert=r_expert, n_min=15, epsilon_hrd=0.15, n_max_leaf=150)
    mw_hrd = ExpertMWUA(hrd, eta=0.5, r_expert=r_expert,
                    candidate_pool_size=12, max_experts=300, combined_basis_dim=r_expert, random_seed=0)
    
    bad_baseline = BadNetBaseline(k, r_expert)
    t = 0
    for x in data_sequence:
        mw_hrd.step(x)
        bad_baseline.step(x)
        t += 1
        if (t % 100 == 0):
            print(t)
    
    return {
        'hrd_cumulative': mw_hrd.cum_loss[1:],
        'bad_cumulative': bad_baseline.cumulative_loss[1:],
        'num_leaves': len(hrd.leaves)
    }

def plot_results(optimal_results, comparison_results):
    """Create plots comparing the algorithms"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Optimal data experiment
    ax1 = axes[0, 0]
    ax1.plot(optimal_results['hrd_cumulative'], label='HRD Algorithm', linewidth=2)
    ax1.plot(optimal_results['bad_cumulative'], label='Bad Net', linewidth=2, linestyle='--')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Loss')
    ax1.set_title('Optimal Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Clustered data
    ax2 = axes[0, 1]
    ax2.plot(comparison_results['clustered']['hrd_cumulative'], label='HRD Algorithm', linewidth=2)
    ax2.plot(comparison_results['clustered']['bad_cumulative'], label='Bad Net', linewidth=2, linestyle='--')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Loss')
    ax2.set_title('Clustered Data')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Random data
    ax3 = axes[1, 0]
    ax3.plot(comparison_results['random']['hrd_cumulative'], label='HRD Algorithm', linewidth=2)
    ax3.plot(comparison_results['random']['bad_cumulative'], label='Bad Net', linewidth=2, linestyle='--')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Cumulative Loss')
    ax3.set_title('Random Data')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Remove the fourth subplot
    axes[1, 1].remove()
    
    plt.tight_layout()
    plt.show()
    
    # Summary plot: all experiments on one graph
    plt.figure(figsize=(10, 6))
    
    # Plot final cumulative loss ratios
    datasets = ['Optimal 2D', 'Clustered', 'Random']
    hrd_final = [
        optimal_results['hrd_cumulative'][-1],
        comparison_results['clustered']['hrd_cumulative'][-1],
        comparison_results['random']['hrd_cumulative'][-1]
    ]
    bad_final = [
        optimal_results['bad_cumulative'][-1],
        comparison_results['clustered']['bad_cumulative'][-1],
        comparison_results['random']['bad_cumulative'][-1]
    ]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, hrd_final, width, label='HRD Algorithm', alpha=0.8)
    plt.bar(x + width/2, bad_final, width, label='Bad Net', alpha=0.8)
    
    plt.xlabel('Dataset Type')
    plt.ylabel('Final Cumulative Loss')
    plt.title('Final Performance Comparison')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add improvement percentages
    for i in range(len(datasets)):
        improvement = (bad_final[i] - hrd_final[i]) / bad_final[i] * 100
        plt.text(i, max(hrd_final[i], bad_final[i]) + 0.1, f'{improvement:.1f}% better', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def print_summary(optimal_results, comparison_results):
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nOptimal 2D Subspace Experiment:")
    print(f"  HRD Final Loss: {optimal_results['hrd_cumulative'][-1]:.4f}")
    print(f"  Bad Net Final Loss: {optimal_results['bad_cumulative'][-1]:.4f}")
    improvement = (optimal_results['bad_cumulative'][-1] - optimal_results['hrd_cumulative'][-1]) / optimal_results['bad_cumulative'][-1] * 100
    print(f"  HRD Improvement: {improvement:.2f}%")
    print(f"  Number of Leaves Created: {optimal_results['num_leaves']}")
    
    for dataset in ['clustered', 'random']:
        result = comparison_results[dataset]
        print(f"\n{dataset.capitalize()} Data:")
        print(f"  HRD Final Loss: {result['hrd_cumulative'][-1]:.4f}")
        print(f"  Bad Net Final Loss: {result['bad_cumulative'][-1]:.4f}")
        improvement = (result['bad_cumulative'][-1] - result['hrd_cumulative'][-1]) / result['bad_cumulative'][-1] * 100
        print(f"  HRD Improvement: {improvement:.2f}%")
        print(f"  Number of Leaves Created: {result['num_leaves']}")

def test_performance_benchmark():
    """Run the complete benchmark experiment"""
    print("Starting HRD Algorithm Performance Benchmark")
    print("=" * 50)
    
    np.random.seed(42)
    
    optimal_results = run_optimal_data_experiment()
    print("DONE with optimal")
    comparison_results = run_comparison_experiment()
    print("DONE with comparison")
    
    print_summary(optimal_results, comparison_results)
    
    print("\nGenerating plots...")
    plot_results(optimal_results, comparison_results)
    
    print("\nBenchmark completed!")
    return optimal_results, comparison_results

if __name__ == "__main__":

    test_performance_benchmark()