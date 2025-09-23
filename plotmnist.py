import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_mnist_cumulative_comparison():
    
    files = [
        'mnist_data_k10_results.csv',
        'mnist_data_k15_results.csv', 
        'mnist_data_k20_results.csv'
    ]
    
    data = {}
    for file in files:
        try:
            df = pd.read_csv(file)
            k_value = df['k_dimension'].iloc[0]
            data[k_value] = df
            print(f"Loaded {file}: {len(df)} time steps")
        except FileNotFoundError:
            print(f"{file} not found")
            continue
    
    if not data:
        print("No CSV files found")
        return
    
    k_values = sorted(data.keys())
    print(f"Found data for k values: {k_values}")
    
    for k in k_values:
        df = data[k]
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['time_step'], df['hrd_cumulative_loss'], 
                label='HRD Algorithm', linewidth=2, color='blue')
        plt.plot(df['time_step'], df['badnet_cumulative_loss'], 
                label='Fixed Baseline', linewidth=2, color='red', linestyle='--')
        
        plt.xlabel('Time Step', fontsize=22)
        plt.ylabel('Cumulative Loss', fontsize=22)
        plt.title(f'MNIST Data - Algorithm Performance (k={k})', fontsize=22)
        plt.legend(fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'mnist_k{k}_cumulative_loss.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")

if __name__ == "__main__":
    plot_mnist_cumulative_comparison()