from final_research_c import *
from synexperiments import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, defaultdict

from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_openml
from collections import defaultdict
import csv


def normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    return X / norms

# COMMENTED OUT - MNIST tests removed
# def run_MNIST_data(k=20):
#     print(f"Running MNIST with k={k}...")
#     mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#     X = mnist['data']
#     y = mnist['target'].astype(int)
# 
#     print("MNIST shape:", X.shape)  # 70000 by 784
# 
#     d_split = min(15, k-1)
#     r_expert = 10
#     print(f"Reducing to {k} dimensions with TruncatedSVD...")
#     svd = TruncatedSVD(n_components=k, random_state=42)
#     X_reduced = svd.fit_transform(X)
# 
#     X_unit = normalize_rows(X_reduced)
# 
#     hrd = SphericalHRD(k=k, d_split=d_split, r_expert=r_expert, n_min=20, epsilon_hrd=0.1, n_max_leaf=100)
#     mw = ExpertMWUA(hrd, eta=0.5, r_expert=r_expert,
#                     candidate_pool_size=12, max_experts=300, combined_basis_dim=r_expert, random_seed=0)
#     
#     badnet = BadNetBaseline(k=k, r=r_expert) #this is the Fixed Baseline generated in experiments class 
#     
#     hrd_losses = []
#     badnet_losses = []
#     
#     for i, x in enumerate(X_unit[:500]):  # limit to 500 for speed
#         agg_loss, chosen, basis = mw.step(x)
#         hrd_losses.append(agg_loss)
#         
#         badnet_loss = badnet.step(x)
#         badnet_losses.append(badnet_loss)
#         
#         if (i+1) % 25 == 0:
#             print(f"  Step {i+1}")
#     
#     return {
#         'hrd_cumulative': mw.cum_loss[1:],
#         'hrd_instantaneous': hrd_losses,
#         'badnet_cumulative': badnet.cumulative_loss[1:],
#         'badnet_instantaneous': badnet_losses,
#         'num_leaves': len(hrd.leaves)
#     }

def run_Credit_Card_data(k=28, r_expert=2):  # Updated to accept r_expert parameter
    print(f"Running Credit Card with k={k}, r_expert={r_expert}...")
    file_path = "creditcard.csv"
    data_list = []
    
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
    original_k = X.shape[1]
    
    # Apply dimensionality reduction if k is specified and different from original
    if k is not None and k != original_k:
        print(f"Reducing from {original_k} to {k} dimensions with TruncatedSVD...")
        svd = TruncatedSVD(n_components=k, random_state=42)
        X = svd.fit_transform(X)
    else:
        k = original_k
    
    d_split = min(15, k-1)
    
    X_unit = normalize_rows(X)
    
    hrd = SphericalHRD(k=k, d_split=d_split, r_expert=r_expert, n_min=20, epsilon_hrd=0.1, n_max_leaf=100)
    mw = ExpertMWUA(hrd, eta=0.5, r_expert=r_expert,
                    candidate_pool_size=12, max_experts=300, combined_basis_dim=r_expert, random_seed=0)
        
    badnet = BadNetBaseline(k=k, r=r_expert)
    
    hrd_losses = []
    badnet_losses = []
    
    print("Streaming Credit Card Data vectors into both algorithms...")
    for i, x in enumerate(X_unit[:500]):
        agg_loss, chosen, basis = mw.step(x)
        hrd_losses.append(agg_loss)
        
        badnet_loss = badnet.step(x)
        badnet_losses.append(badnet_loss)
        
        if (i+1) % 25 == 0:
            print(f"  Step {i+1}")
    
    return {
        'hrd_cumulative': mw.cum_loss[1:],
        'hrd_instantaneous': hrd_losses,
        'badnet_cumulative': badnet.cumulative_loss[1:],
        'badnet_instantaneous': badnet_losses,
        'num_leaves': len(hrd.leaves)
    }

def run_experiments_multiple_r_expert():
    """Run experiments with different r_expert values for credit card dataset"""
    
    k = 28  # Fixed k value for credit card data
    r_expert_values = [10, 15, 20]  # Test different r_expert values
    
    results = {
        'creditcard': {}
    }
    
    np.random.seed(42) 
    
    for r_expert in r_expert_values:
        print(f"\n{'='*50}")
        print(f"Running Credit Card experiment with k={k}, r_expert={r_expert}")
        print(f"{'='*50}")
        
        results['creditcard'][r_expert] = run_Credit_Card_data(k=k, r_expert=r_expert)
        current_results = {'creditcard': {r_expert: results['creditcard'][r_expert]}}
        save_results_to_csv(current_results, [r_expert])
    return results

def plot_single_dataset_experiment(results_dict, dataset_name, r_expert_value):
    """Plot results for a single dataset and r_expert value"""
    result = results_dict[r_expert_value]
    
    plt.figure(figsize=(10, 6))
    plt.plot(result['hrd_cumulative'], label='HRD Algorithm', linewidth=2, color='blue')
    plt.plot(result['badnet_cumulative'], label='Fixed Baseline', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Loss')
    plt.title(f'{dataset_name.title()} Data - Algorithm Performance (k=28, r_expert={r_expert_value})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{dataset_name}_data_r{r_expert_value}_performance.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return filename

def save_results_to_csv(all_results, r_expert_values):
    """Save experimental results to CSV files"""
    print("\nSaving results to CSV files...")
    
    datasets = ['creditcard']  # Only credit card now
    saved_files = []
    
    for dataset in datasets:
        for r_expert in r_expert_values:
            result = all_results[dataset][r_expert]
            
            df = pd.DataFrame({
                'time_step': range(len(result['hrd_cumulative'])),
                'hrd_cumulative_loss': result['hrd_cumulative'],
                'badnet_cumulative_loss': result['badnet_cumulative'],
                'hrd_instantaneous_loss': result['hrd_instantaneous'],
                'badnet_instantaneous_loss': result['badnet_instantaneous']
            })
            
            df['dataset'] = dataset
            df['k_dimension'] = 28  # Fixed k
            df['r_expert'] = r_expert  # Variable r_expert
            df['num_leaves'] = result['num_leaves']
            
            df['improvement_percentage'] = ((df['badnet_cumulative_loss'] - df['hrd_cumulative_loss']) / df['badnet_cumulative_loss'] * 100)
            
            filename = f'{dataset}_data_r{r_expert}_results.csv'
            df.to_csv(filename, index=False)
            saved_files.append(filename)
            
            print(f"  Saved: {filename}")
    
    summary_data = []
    for dataset in datasets:
        for r_expert in r_expert_values:
            result = all_results[dataset][r_expert]
            summary_data.append({
                'dataset': dataset,
                'k_dimension': 28,  # Fixed k
                'r_expert': r_expert,  # Variable r_expert
                'hrd_final_loss': result['hrd_cumulative'][-1],
                'badnet_final_loss': result['badnet_cumulative'][-1],
                'improvement_percentage': ((result['badnet_cumulative'][-1] - result['hrd_cumulative'][-1]) / result['badnet_cumulative'][-1] * 100),
                'num_leaves': result['num_leaves'],
                'total_time_steps': len(result['hrd_cumulative'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = 'creditcard_experiment_summary.csv'
    summary_df.to_csv(summary_filename, index=False)
    saved_files.append(summary_filename)
    print(f"  Saved: {summary_filename}")
    
    return saved_files

def print_detailed_summary(all_results, r_expert_values):
    print("\n" + "="*80)
    print("CREDIT CARD FRAUD DETECTION EXPERIMENT SUMMARY")
    print("Fixed k=28, varying r_expert values")
    print("="*80)
    
    dataset = 'creditcard'
    print(f"\nCredit Card Fraud Detection Data:")
    print("-" * 35)
    
    for r_expert in r_expert_values:
        result = all_results[dataset][r_expert]
        hrd_final = result['hrd_cumulative'][-1]
        bad_final = result['badnet_cumulative'][-1]
        improvement = (bad_final - hrd_final) / bad_final * 100
        
        print(f"  r_expert={r_expert}:")
        print(f"    HRD Final Loss: {hrd_final:.4f}")
        print(f"    Fixed Baseline Final Loss: {bad_final:.4f}")
        print(f"    HRD Improvement: {improvement:.2f}%")
        print(f"    Number of Leaves: {result['num_leaves']}")
        print()

def test_performance_benchmark_multiple_r_expert():
    print("Testing Credit Card Fraud Detection with multiple r_expert values")
    print("Fixed k=28, varying r_expert=[10, 15, 20]")
    print("=" * 60)
    
    all_results = run_experiments_multiple_r_expert()
    r_expert_values = [10, 15, 20]
    
    print_detailed_summary(all_results, r_expert_values)
    
    print("\nSaving data to CSV files...")
    saved_csv_files = save_results_to_csv(all_results, r_expert_values)
    
    print("\nGenerating individual performance plots...")
    generated_files = []
    
    dataset = 'creditcard'
    for r_expert in r_expert_values:
        filename = plot_single_dataset_experiment(all_results[dataset], dataset, r_expert)
        generated_files.append(filename)
    
    print(f"\nGenerated {len(saved_csv_files)} CSV files:")
    for filename in saved_csv_files:
        print(f"  - {filename}")
    
    print(f"\nGenerated {len(generated_files)} plot files:")
    for filename in generated_files:
        print(f"  - {filename}")
    
    print("\nCredit Card Fraud Detection Benchmark completed!")
    return all_results

if __name__ == "__main__":
    test_performance_benchmark_multiple_r_expert()