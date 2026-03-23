import numpy as np
import torch
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from holo_cpu import HoloRISC_LSTM, int_to_16bit_vec

def generate_dataset(num_samples=5000): # Reduced for speed
    print(f"Generating {num_samples} samples of CPU behavior...")
    cpu = HoloRISC_LSTM()
    
    inputs = []
    outputs = []
    
    for i in range(num_samples):
        try:
            # 1. Randomize State
            for j in range(4):
                cpu.R[j] = np.random.randint(0, 2, 8).astype(complex)
            cpu.PC = np.random.randint(0, 2, 8).astype(complex)
            cpu.PH = complex(float(np.random.randint(0, 2)))
            
            # 2. Random Instruction (16 bits)
            op_choices = [1, 2, 3, 15]
            op = np.random.choice(op_choices)
            rd = np.random.randint(0, 4)
            rs = np.random.randint(0, 4)
            imm = np.random.randint(0, 256)
            
            instr_val = (op << 12) | (rd << 10) | (rs << 8) | imm
            instr_vec = int_to_16bit_vec(instr_val)
            cpu.IR = instr_vec.copy()
            
            state_in = np.concatenate([
                instr_vec.real,
                cpu.R[0].real, cpu.R[1].real, cpu.R[2].real, cpu.R[3].real,
                cpu.PC.real,
                [cpu.PH.real]
            ])
            
            # 3. Step CPU
            cpu.step()
            
            state_out = np.concatenate([
                cpu.R[0].real, cpu.R[1].real, cpu.R[2].real, cpu.R[3].real,
                cpu.PC.real,
                [cpu.PH.real]
            ])
            
            inputs.append(state_in)
            outputs.append(state_out)
            
            if (i+1) % 1000 == 0:
                print(f"Progress: {i+1}/{num_samples}")
                
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            break
        
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    save_path = "results/cpu_behavior_data.npz"
    np.savez(save_path, inputs=inputs, outputs=outputs)
    print(f"Dataset saved to {save_path}")
    print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}")

if __name__ == "__main__":
    generate_dataset()

