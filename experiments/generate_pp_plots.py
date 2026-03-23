import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_pp_evolution_plot():
    # Use the newly generated PP data
    csv_path = "c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/Network/experiments/results/pp_emergence_log.csv"
    if not os.path.exists(csv_path):
        print(f" Error: {csv_path} not found. Please run train_pp_emergence.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Set style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Phase Transition Plot: Identity (Acc) vs Kb
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Global Training Step', fontsize=12)
    ax1.set_ylabel('Basin-Captured Identity Rate (Acc)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(df['Step'], df['Perfect_Bit_Identity_Rate'], color=color, linewidth=3, label='Identity Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Bit-Weight Kinetic Energy (Kb)', color=color, fontsize=12, fontweight='bold')
    # Filter out initial spike if it's too large for visualization (optional, but keep for consistency)
    ax2.plot(df['Step'], df['Bit_Weight_Kinetic_Energy_Kb'], color=color, linewidth=2, linestyle='--', marker='o', markersize=4, label='Kinetic Energy (Kb)', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Find the jump point: KB陡然下降, ACC陡然上升
    # Step 0 -> Step 10 is the clear jump in the data
    emergence_step = 10
    jump_row = df[df['Step'] == emergence_step].iloc[0]
    start_rate = df[df['Step'] == 0]['Perfect_Bit_Identity_Rate'].iloc[0]
    end_rate = jump_row['Perfect_Bit_Identity_Rate']
    
    ax1.axvline(x=emergence_step, color='green', linestyle=':', alpha=0.8, linewidth=2)
    
    # Annotate the jump
    ax1.annotate(f'Fracture Jump\n({start_rate:.1%} -> {end_rate:.1%})', 
                 xy=(emergence_step, end_rate), xytext=(emergence_step + 150, end_rate + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=10, color='darkgreen', fontweight='bold')
    
    plt.title('Logic Phase Transition in Real-Language Predictive Tasks', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    output_path = "c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/Paper/pp_logic_phase_transition.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Successfully generated {output_path} based on real-language PP tasks.")

if __name__ == "__main__":
    generate_pp_evolution_plot()
