import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_synthetic_cumulative_comparison():
    """Load synthetic CSV files and create cumulative loss comparison plots"""
    
    datasets = ['optimal', 'clustered']
    k_values = [2, 5, 10]
    
    data = {}
    for dataset in datasets:
        data[dataset] = {}
        for k in k_values:
            filename = f'{dataset}_data_r{k}_results.csv'
            try:
                df = pd.read_csv(filename)
                data[dataset][k] = df
                print(f"Loaded {filename}: {len(df)} time steps")
            except FileNotFoundError:
                print(f"{filename} not found")
                continue
    
    for dataset in datasets:
        for k in k_values:
            if k not in data[dataset]:
                continue
                
            df = data[dataset][k]
            
            plt.figure(figsize=(10, 6))
            plt.plot(df['time_step'], df['hrd_cumulative_loss'], 
                    label='HRD Algorithm', linewidth=2, color='blue')
            plt.plot(df['time_step'], df['bad_cumulative_loss'], 
                    label='Fixed Baseline', linewidth=2, color='red', linestyle='--')
            
            plt.xlabel('Time Step', fontsize=22)
            plt.ylabel('Cumulative Loss', fontsize=22)
            plt.title(f'{dataset.title()} Data - Algorithm Performance (k={k})', fontsize=22)
            plt.legend(fontsize=18)
            plt.tick_params(axis='both', which='major', labelsize=18)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f'synthetic_{dataset}_k{k}_cumulative_loss.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Saved: {filename}")
    
    
    print("\n" + "="*60)
    print("SYNTHETIC DATA RESULTS SUMMARY")
    print("="*60)
    
    for dataset in datasets:
        print(f"\n{dataset.title()} Data:")
        print("-" * (len(dataset) + 5))
        
        for k in k_values:
            if k in data[dataset]:
                df = data[dataset][k]
                hrd_final = df['hrd_cumulative_loss'].iloc[-1]
                bad_final = df['bad_cumulative_loss'].iloc[-1]
                improvement = df['improvement_percentage'].iloc[-1]
                
                print(f"  k={k}:")
                print(f"    HRD Final Loss: {hrd_final:.4f}")
                print(f"    Fixed Baseline Final Loss: {bad_final:.4f}")
                print(f"    Final Improvement: {improvement:.2f}%")
                print(f"    Number of Leaves: {df['num_leaves'].iloc[0]}")
                print(f"    Time Steps: {len(df)}")
                print()

if __name__ == "__main__":
    plot_synthetic_cumulative_comparison()