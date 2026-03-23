import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from holo_cpu import HoloLSTM_CPU # type: ignore

def int_to_8bit_vec(val):
    return np.array([(val >> i) & 1 for i in range(8)], dtype=complex)

def int_to_4bit_vec(val):
    return np.array([(val >> i) & 1 for i in range(4)], dtype=complex)

def run_noisy_cpu(noise_level):
    """Runs Fibonacci on HoloCPU with injected complex noise."""
    cpu = HoloLSTM_CPU()
    # Simple Program: LDA 10, ADD 11, STA 11, HLT (Calculates 1+1=2)
    prog = [0xA1, 0xB3, 0xB2, 0x0F] 
    for i, instr in enumerate(prog):
        cpu.rom_data[i] = int_to_8bit_vec(instr)
    cpu.ram[10] = int_to_4bit_vec(1)
    cpu.ram[11] = int_to_4bit_vec(1)
    
    steps = 0
    while cpu.halted.real < 0.5 and steps < 100:
        # Inject Noise at each step
        cpu.ACC += (np.random.randn(4) + 1j*np.random.randn(4)) * noise_level
        cpu.PC += (np.random.randn(4) + 1j*np.random.randn(4)) * noise_level
        cpu.step()
        steps += 1
        
        # If any value is NaN or Inf, it's a collapse
        if np.any(np.isnan(cpu.ACC)) or np.any(np.isinf(cpu.ACC)):
            return False
            
    final_val = int(sum(round(np.clip(cpu.ram[11][j].real, 0, 1)) * (2**j) for j in range(4)))
    return final_val == 2

def perform_robustness_scan():
    print("--- HoloCPU Robustness Scan: Logic Collapse Threshold ---")
    noise_levels = np.logspace(-6, -0.3, 20)
    success_rates = []
    
    for nl in noise_levels:
        successes = 0
        trials = 50 # Increased for better manifold resolution
        for _ in range(trials):
            if run_noisy_cpu(nl):
                successes += 1
        rate = successes / trials
        success_rates.append(rate)
        print(f"Noise Level: {nl:.2e} | Success Rate: {rate:.2%}")

    # Save data for Nature figure
    df = pd.DataFrame({"noise_level": noise_levels, "success_rate": success_rates})
    os.makedirs("../results", exist_ok=True)
    df.to_csv("../results/robustness_manifold_data.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(noise_levels, success_rates, 'b-o', linewidth=2)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Boolean Threshold (0.5)')
    plt.title("HoloCPU Logic Robustness vs. Complex Noise")
    plt.xlabel("Noise Amplitude (Std Dev)")
    plt.ylabel("Success Rate (Fibonacci Correctness)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig("../results/robustness_results.png")
    print("\nScan complete. Results saved to results/robustness_results.png")

if __name__ == "__main__":
    perform_robustness_scan()
