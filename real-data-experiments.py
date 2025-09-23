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

def run_MNIST_data(k=20):
    print(f"Running MNIST with k={k}...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data']
    y = mnist['target'].astype(int)

    print("MNIST shape:", X.shape)

    d_split = min(15, k-1)
    r_expert = 10
    print(f"Reducing to {k} dimensions with TruncatedSVD...")
    svd = TruncatedSVD(n_components=k, random_state=42)
    X_reduced = svd.fit_transform(X)

    X_unit = normalize_rows(X_reduced)

    hrd = SphericalHRD(k=k, d_split=d_split, r_expert=r_expert, n_min=20, epsilon_hrd=0.1, n_max_leaf=100)
    mw = ExpertMWUA(hrd, eta=0.5, r_expert=r_expert,
                    candidate_pool_size=12, max_experts=300, combined_basis_dim=r_expert, random_seed=0)
    
    badnet = BadNetBaseline(k=k, r=r_expert)
    
    hrd_losses = []
    badnet_losses = []
    
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

def run_Credit_Card_data(k=None):
    print(f"Running Credit Card with k={k if k else 'original'}...")
    file_path = "creditcard.csv"
    data_list = []
    
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        
        print("Header:", header)
        print("Total columns:", len(header))
        
        columns_to_remove = [0, len(header)-2, len(header)-1]
        
        for row in reader:
            pruned_row = [float(row[i]) for i in range(len(row)) if i not in columns_to_remove]
            data_list.append(pruned_row)
    
    X = np.array(data_list)
    original_k = X.shape[1]
    
    if k is not None and k != original_k:
        print(f"Reducing from {original_k} to {k} dimensions with TruncatedSVD...")
        svd = TruncatedSVD(n_components=k, random_state=42)
        X = svd.fit_transform(X)
    else:
        k = original_k
    
    d_split = min(15, k-1)
    r_expert = 2
    
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

def run_experiments_multiple_k():
    k_values = [10, 15, 20]  
    
    results = {
        'mnist': {},
        'creditcard': {}
    }
    
    np.random.seed(42) 
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Running experiments with k={k}")
        print(f"{'='*50}")
        
        print(f"MNIST data experiment (k={k})...")
        results['mnist'][k] = run_MNIST_data(k)
        
        print(f"Credit Card data experiment (k={k})...")
        results['creditcard'][k] = run_Credit_Card_data(k)
    
    return results

def save_results_to_csv(all_results, k_values):
    print("\nSaving results to CSV files...")
    
    datasets = ['mnist', 'creditcard']
    saved_files = []
    
    for dataset in datasets:
        for k in k_values:
            result = all_results[dataset][k]
            
            df = pd.DataFrame({
                'time_step': range(len(result['hrd_cumulative'])),
                'hrd_cumulative_loss': result['hrd_cumulative'],
                'badnet_cumulative_loss': result['badnet_cumulative'],
                'hrd_instantaneous_loss': result['hrd_instantaneous'],
                'badnet_instantaneous_loss': result['badnet_instantaneous']
            })
            
            df['dataset'] = dataset
            df['k_dimension'] = k
            df['num_leaves'] = result['num_leaves']
            
            df['improvement_percentage'] = ((df['badnet_cumulative_loss'] - df['hrd_cumulative_loss']) / df['badnet_cumulative_loss'] * 100)
            
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
                'badnet_final_loss': result['badnet_cumulative'][-1],
                'improvement_percentage': ((result['badnet_cumulative'][-1] - result['hrd_cumulative'][-1]) / result['badnet_cumulative'][-1] * 100),
                'num_leaves': result['num_leaves'],
                'total_time_steps': len(result['hrd_cumulative'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = 'real_data_experiment_summary.csv'
    summary_df.to_csv(summary_filename, index=False)
    saved_files.append(summary_filename)
    print(f"  Saved: {summary_filename}")
    
    return saved_files

def print_detailed_summary(all_results, k_values):
    print("\n" + "="*80)
    print("REAL DATA EXPERIMENT SUMMARY")
    print("="*80)
    
    datasets = ['mnist', 'creditcard']
    dataset_labels = ['MNIST Data', 'Credit Card Data']
    
    for dataset, label in zip(datasets, dataset_labels):
        print(f"\n{label}:")
        print("-" * len(label))
        
        for k in k_values:
            result = all_results[dataset][k]
            hrd_final = result['hrd_cumulative'][-1]
            bad_final = result['badnet_cumulative'][-1]
            improvement = (bad_final - hrd_final) / bad_final * 100
            
            print(f"  k={k}:")
            print(f"    HRD Final Loss: {hrd_final:.4f}")
            print(f"    Fixed Baseline Final Loss: {bad_final:.4f}")
            print(f"    HRD Improvement: {improvement:.2f}%")
            print(f"    Number of Leaves: {result['num_leaves']}")
            print()

def test_performance_benchmark_multiple_k():
    print("Testing multiple dimensionalities (k values)")
    print("=" * 60)
    
    all_results = run_experiments_multiple_k()
    k_values = [10, 15, 20]
    
    print_detailed_summary(all_results, k_values)
    
    print("\nSaving data to CSV files...")
    saved_csv_files = save_results_to_csv(all_results, k_values)
    
    print(f"\nGenerated {len(saved_csv_files)} CSV files:")
    for filename in saved_csv_files:
        print(f"  - {filename}")
    
    print("\nReal Data Benchmark completed!")
    return all_results

if __name__ == "__main__":
    test_performance_benchmark_multiple_k()