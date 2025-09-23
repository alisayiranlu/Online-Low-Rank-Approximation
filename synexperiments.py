from final_research_c import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, defaultdict

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

class BadNetBaseline:
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
    
    def step(self, x):
        loss = projection_loss(self.basis, x)
        self.cumulative_loss.append(self.cumulative_loss[-1] + loss)
        return loss

def generate_optimal_data_sequence(T, k=5):
    sequence = []
    for t in range(T):
        x1 = np.random.randn()
        x2 = np.random.randn()
        x = np.zeros(k)
        x[0] = x1
        x[1] = x2
        
        x_normalized = normalize(x)
        sequence.append(x_normalized)
    
    return sequence

def generate_clustered_data_sequence(T, k=5):
    clustered_data = []
    centers = [normalize(np.random.randn(k)) for _ in range(3)]
    for t in range(T):
        center = centers[t % 3]
        noise = 0.3 * np.random.randn(k)
        x = normalize(center + noise)
        clustered_data.append(x)
    return clustered_data

def run_single_experiment(data_sequence, k, r_expert=2):
    d_split = min(4, k-1) 
    
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
            print(f"  Step {t}")
    
    return {
        'hrd_cumulative': mw_hrd.cum_loss[1:],
        'bad_cumulative': bad_baseline.cumulative_loss[1:],
        'hrd_instantaneous': hrd_losses,
        'bad_instantaneous': bad_losses,
        'num_leaves': len(hrd.leaves)
    }

def run_experiments_multiple_k():
    T = 1000
    k_values = [5, 10, 15]  
    r_expert = 2
    
    results = {
        'optimal': {},
        'clustered': {}
    }
    
    np.random.seed(42)  
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Running experiments with k={k}")
        print(f"{'='*50}")
        
        print(f"Optimal data experiment (k={k})...")
        optimal_data = generate_optimal_data_sequence(T, k)
        results['optimal'][k] = run_single_experiment(optimal_data, k, r_expert)
        
        print(f"Clustered data experiment (k={k})...")
        clustered_data = generate_clustered_data_sequence(T, k)
        results['clustered'][k] = run_single_experiment(clustered_data, k, r_expert)
    
    return results

def save_results_to_csv(all_results, k_values):
    print("\nSaving results to CSV files...")
    
    datasets = ['optimal', 'clustered']
    saved_files = []
    
    for dataset in datasets:
        for k in k_values:
            result = all_results[dataset][k]
            
            df = pd.DataFrame({
                'time_step': range(len(result['hrd_cumulative'])),
                'hrd_cumulative_loss': result['hrd_cumulative'],
                'bad_cumulative_loss': result['bad_cumulative'],
                'hrd_instantaneous_loss': result['hrd_instantaneous'],
                'bad_instantaneous_loss': result['bad_instantaneous']
            })
            
            df['dataset'] = dataset
            df['k_dimension'] = k
            df['num_leaves'] = result['num_leaves']
            
            df['improvement_percentage'] = ((df['bad_cumulative_loss'] - df['hrd_cumulative_loss']) / df['bad_cumulative_loss'] * 100)
            
            filename = f'{dataset}_data_k{k}_results.csv'
            df.to_csv(filename, index=False)
            saved_files.append(filename)
            print(f"  Saved: {filename}")
    
    summary_data = []
    for dataset in datasets:
        for k in k_values:
            result = all_results[dataset][k]
            summary_data.append({
                'dataset': dataset,
                'k_dimension': k,
                'hrd_final_loss': result['hrd_cumulative'][-1],
                'bad_final_loss': result['bad_cumulative'][-1],
                'improvement_percentage': ((result['bad_cumulative'][-1] - result['hrd_cumulative'][-1]) / result['bad_cumulative'][-1] * 100),
                'num_leaves': result['num_leaves'],
                'total_time_steps': len(result['hrd_cumulative'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = 'experiment_summary.csv'
    summary_df.to_csv(summary_filename, index=False)
    saved_files.append(summary_filename)
    print(f"  Saved: {summary_filename}")
    
    return saved_files

def print_detailed_summary(all_results, k_values):
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*80)
    
    datasets = ['optimal', 'clustered']
    dataset_labels = ['Optimal 2D Subspace', 'Clustered Data']
    
    for dataset, label in zip(datasets, dataset_labels):
        print(f"\n{label}:")
        print("-" * len(label))
        
        for k in k_values:
            result = all_results[dataset][k]
            hrd_final = result['hrd_cumulative'][-1]
            bad_final = result['bad_cumulative'][-1]
            improvement = (bad_final - hrd_final) / bad_final * 100
            
            print(f"  k={k}:")
            print(f"    HRD Final Loss: {hrd_final:.4f}")
            print(f"    Bad Net Final Loss: {bad_final:.4f}")
            print(f"    HRD Improvement: {improvement:.2f}%")
            print(f"    Number of Leaves: {result['num_leaves']}")
            print()
   
def test_performance_benchmark_multiple_k():
    print("Starting Enhanced HRD Algorithm Performance Benchmark")
    print("Testing multiple dimensionalities (k values)")
    print("=" * 60)
    
    all_results = run_experiments_multiple_k()
    k_values = [5, 10, 15]
    
    print_detailed_summary(all_results, k_values)
    
    print("\nSaving data to CSV files...")
    saved_csv_files = save_results_to_csv(all_results, k_values)
    
    print(f"\nGenerated {len(saved_csv_files)} CSV files:")
    for filename in saved_csv_files:
        print(f"  - {filename}")
    
    print("\nBenchmark completed!")
    return all_results

if __name__ == "__main__":
    test_performance_benchmark_multiple_k()