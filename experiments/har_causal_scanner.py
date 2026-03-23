import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# 1. Simplified HAR Dataset Loader (Pre-processed version usually has 561 features)
# We will use a mock representative subset if the full dataset is too large/complex for a quick scan.
def load_har_representative_sample():
    # In a real scenario, we'd load the .txt files. 
    # For this "Genesis" evidence, we'll focus on the logical mapping of 6 core axes:
    # Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z
    num_samples = 1000
    X = np.random.randn(num_samples, 10, 6) # 10 time steps, 6 features
    # Create a logical ground truth: if Acc_Z > 1.0 (threshold), it's "Walking"
    Y = (X[:, -1, 2] > 0.5).astype(float) 
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

# 2. Holomorphic-ready LSTM
class HAR_LSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super(HAR_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return self.sigmoid(out)

def csd_decompose(val, fractional_bits=16):
    """Decomposes a float into shift-and-add operations (Bit-Perfect Analysis)."""
    scale = 2**fractional_bits
    if abs(val) < 1e-9: return [], 0
    q_val = int(round(float(val) * scale))
    if q_val == 0: return [], 0
    res, i, temp_q = [], 0, q_val
    max_shift = 0
    while temp_q != 0:
        if temp_q % 2 != 0:
            digit = 2 - (temp_q % 4)
            temp_q -= digit
            res.append(i)
            max_shift = max(max_shift, i)
        temp_q //= 2; i += 1
    return res, max_shift

# 3. Causal Logic Scanner (Hardware-Logic Paradigm)
def scan_har_causality(model):
    print("🔍 Scanning HAR Logic Circuit (ANS Paradigm)...")
    hidden_dim = model.lstm.hidden_size
    weights_ih = model.lstm.weight_ih_l0.data.cpu().numpy() # (4*hidden, input)
    
    # LSTM Gates: i(0:h), f(h:2h), g(2h:3h), o(3h:4h)
    w_i = weights_ih[0:hidden_dim, :]
    w_f = weights_ih[hidden_dim:2*hidden_dim, :]
    w_g = weights_ih[2*hidden_dim:3*hidden_dim, :]
    
    input_names = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
    causal_map = []
    
    for i, name in enumerate(input_names):
        # 1. Control Logic Strength: Contribution to Reset (f) and Write Enable (i)
        control_strength = (np.abs(w_i[:, i]).mean() + np.abs(w_f[:, i]).mean()) / 2.0
        
        # 2. Data Flow Gain: Contribution to the core signal transformation (g)
        data_gain = np.abs(w_g[:, i]).mean()
        
        # 3. CSD MSB Rank: Highest bit-shift level across all gates for this feature
        all_feature_weights = weights_ih[:, i]
        msb_levels = [csd_decompose(w)[1] for w in all_feature_weights]
        max_msb = max(msb_levels) if msb_levels else 0
        
        # Total Isomorphic Weight (Unified measure of circuit importance)
        total_logic_weight = control_strength * 0.7 + data_gain * 0.3
        
        causal_map.append({
            "Feature": name, 
            "Control_Logic": control_strength,
            "Data_Gain": data_gain,
            "MSB_Level": max_msb,
            "Logic_Weight": total_logic_weight
        })
    
    df = pd.DataFrame(causal_map).sort_values(by="Logic_Weight", ascending=False)
    print("\n--- Direct Circuit Reading: Sensor-to-Logic Mapping ---")
    print(df.to_string(index=False))
    return df

def run_har_experiment():
    X, Y = load_har_representative_sample()
    model = HAR_LSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    print("🚀 Training HAR LSTM for logic discovery...")
    for epoch in range(20):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            acc = ((out > 0.5) == (Y > 0.5)).float().mean()
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
            
    # Scan causality with the new Hardware-Logic paradigm
    df_causal = scan_har_causality(model)
    # Save both versions for backward compatibility and analysis
    df_causal.to_csv("../results/har_causal_logic.csv", index=False)
    print("\n[SUCCESS] HAR Logic Circuit mapped to Control and Data paths.")

if __name__ == "__main__":
    run_har_experiment()
