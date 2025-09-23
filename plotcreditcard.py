import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_plot_credit_card_results():
    """Load Credit Card CSV files and create comprehensive plots with large fonts"""
    
    files = [
        'creditcard_data_k10_results.csv',
        'creditcard_data_k15_results.csv', 
        'creditcard_data_k20_results.csv'
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
        print("No CSV files found!")
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
        
        plt.xlabel('Time Step', fontsize=16)
        plt.ylabel('Cumulative Loss', fontsize=16)
        plt.title(f'Credit Card Data - Cumulative Loss (k={k})', fontsize=18)
        plt.legend(fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'creditcard_k{k}_cumulative_loss.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")
    

    print("\n" + "="*60)
    print("CREDIT CARD RESULTS SUMMARY")
    print("="*60)
    
    for k in k_values:
        df = data[k]
        print(f"\nk={k}:")
        print(f"  Final HRD Loss: {df['hrd_cumulative_loss'].iloc[-1]:.4f}")
        print(f"  Final Fixed Baseline Loss: {df['badnet_cumulative_loss'].iloc[-1]:.4f}")
        print(f"  Final Improvement: {df['improvement_percentage'].iloc[-1]:.2f}%")
        print(f"  Number of Leaves: {df['num_leaves'].iloc[0]}")
        print(f"  Time Steps: {len(df)}")

if __name__ == "__main__":
    load_and_plot_credit_card_results()