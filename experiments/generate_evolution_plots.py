import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_evolution_plots():
    csv_path = "c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/LSTM/logic_evolution_steps.csv"
    df = pd.read_csv(csv_path)
    
    # Set style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # 1. Phase Transition Plot: Perfect Identity vs Kb (Step-level)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Global Training Step', fontsize=12)
    ax1.set_ylabel('Perfect Bit-Identity Rate', color=color, fontsize=12, fontweight='bold')
    ax1.plot(df['step'], df['perfect_rate'], color=color, linewidth=3, label='Perfect Bit-Identity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Bit-Weight Kinetic Energy (Kb)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(df['step'], df['kb'], color=color, linewidth=2, linestyle='--', marker='o', markersize=4, label='Kinetic Energy (Kb)', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight the emergence jump
    emergence_step = 45
    ax1.axvline(x=emergence_step, color='green', linestyle=':', alpha=0.8, linewidth=2)
    
    # Annotate the jump
    # Step 30 was ~15%, Step 45 was ~68%
    ax1.annotate('Fracture Jump\n(15% -> 68%)', 
                 xy=(emergence_step, 0.68), xytext=(emergence_step + 50, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=10, color='darkgreen', fontweight='bold')
    
    plt.title('Logic Phase Transition: The Nonlinear Emergence of Circuitry', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig("c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/Paper/logic_phase_transition.png", dpi=300)
    print("✅ Successfully re-generated logic_phase_transition.png with Perfect Identity jump.")

if __name__ == "__main__":
    generate_evolution_plots()
