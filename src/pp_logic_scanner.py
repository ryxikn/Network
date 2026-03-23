import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add current dir to path to import GPTSmall_Fabric
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from train_pp_emergence import GPTSmall_Fabric, download_data

def csd_decompose(val, fractional_bits=24):
    scale = 2**fractional_bits
    if abs(val) < 1e-12: return [], scale
    q_val = int(round(float(val) * scale))
    if q_val == 0: return [], scale
    res, i, temp_q = [], 0, q_val
    while temp_q != 0:
        if temp_q % 2 != 0:
            digit = 2 - (temp_q % 4)
            temp_q -= digit
            res.append(("+ " if digit > 0 else "- ", i))
        temp_q //= 2; i += 1
    return res, scale

class PPLogicScanner:
    def __init__(self, model_path):
        # Load vocab
        text = download_data()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        # Load model
        self.model = GPTSmall_Fabric(self.vocab_size, 128)
        self.state_dict = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

    def find_logic_drivers(self):
        """Analyzes the weight matrix to find triggers for 'Capitalization'."""
        # 1. Embedding Analysis: Which characters are 'close' in the logical space?
        emb = self.state_dict['embedding.weight'].cpu().numpy() # (65, 128)
        
        # 2. Final FC Layer: Which hidden states drive which output tokens?
        fc_out = self.state_dict['fc_out.weight'].cpu().numpy() # (65, 128)
        
        # Find newline '\n' index
        newline_idx = self.stoi['\n']
        
        # Find some capital letters
        capitals = [c for c in self.chars if 'A' <= c <= 'Z']
        cap_indices = [self.stoi[c] for c in capitals]
        
        print("\n" + "="*80)
        print(" [PP LOGIC CIRCUIT SCANNER] - TINY SHAKESPEARE ")
        print("="*80)
        
        # 3. Direct Causality: Input '\n' -> Output 'A-Z'
        # In a 1-layer simplification, the direct drive is roughly Emb @ FC_Out.T
        causality = np.dot(emb, fc_out.T) # (65, 65)
        
        newline_drive = causality[newline_idx]
        top_drives = np.argsort(newline_drive)[::-1][:10]
        
        print(f"\n[1] TOP TOKENS DRIVEN BY NEWLINE (\\n):")
        for idx in top_drives:
            print(f"    - '{self.itos[idx]}': Strength {newline_drive[idx]:.4f}")
            
        # 4. CSD Decomposition of the 'Newline -> Capital' Gate
        # Pick 'C' as an example (common speaker start)
        target_char = 'C'
        if target_char in self.stoi:
            target_idx = self.stoi[target_char]
            gate_weight = newline_drive[target_idx]
            csd, scale = csd_decompose(gate_weight)
            print(f"\n[2] CIRCUIT SYNTHESIS (CSD) FOR '\\n' -> '{target_char}':")
            print(f"    - Weight: {gate_weight:.8f}")
            print(f"    - CSD: {' '.join([f'{d}2^{p}' for d, p in csd])} / 2^{int(np.log2(scale))}")

if __name__ == "__main__":
    base_path = r"c:\Users\ukiyo\OneDrive\Desktop\transformer\mamba\Network\checkpoints"
    scanner = PPLogicScanner(os.path.join(base_path, "pp_gpt_small_final.pth"))
    scanner.find_logic_drivers()
