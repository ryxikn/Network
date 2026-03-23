import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import argparse
import urllib.request

# --- Universal Neural Logic Fabric (GPT-Small Architecture) ---
class GPTSmall_Fabric(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_heads=4, num_layers=4):
        super(GPTSmall_Fabric, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim*4, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        b, t = x.size()
        x = self.embedding(x) + self.pos_embedding[:, :t, :]
        # Causal mask for Next-Token Prediction (PP Task)
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.fc_out(x)

def download_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = "tinyshakespeare.txt"
    if not os.path.exists(file_path):
        print(f"Downloading TinyShakespeare from {url}...")
        urllib.request.urlretrieve(url, file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def train_pp_emergence(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load Real Language Data (TinyShakespeare) ---
    text = download_data()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    # Simple split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - args.seq_len, (args.batch_size,))
        x = torch.stack([d[i:i+args.seq_len] for i in ix])
        y = torch.stack([d[i+1:i+args.seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    model = GPTSmall_Fabric(vocab_size, args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    history = {"epoch": [], "identity_rate": [], "kb": [], "loss": []}
    prev_weights = {k: v.clone() for k, v in model.state_dict().items()}
    
    print(f"Starting Real-Language PP-Emergence Training | Vocab: {vocab_size} | Device: {device}")

    # We use 'Step' as epoch for granularity in the plot
    for step in range(args.steps):
        model.train()
        xb, yb = get_batch('train')
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        optimizer.step()

        # --- Calculate Bit-Weight Kinetic Energy (Kb) ---
        kb = 0
        curr_weights = model.state_dict()
        for k in curr_weights:
            kb += torch.sum((curr_weights[k] - prev_weights[k])**2).item()
        prev_weights = {k: v.clone() for k, v in curr_weights.items()}

        # --- Evaluate Basin-wise Identity Rate (Top-1 Accuracy) ---
        if step % 10 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch('val')
                v_logits = model(xv)
                preds = torch.argmax(v_logits, dim=-1)
                identity_rate = (preds == yv).float().mean().item()
            
            history["epoch"].append(step)
            history["loss"].append(loss.item())
            history["identity_rate"].append(identity_rate)
            history["kb"].append(kb)

            if (step % 50 == 0):
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | Identity (Acc): {identity_rate:.4f} | Kb: {kb:.4e}")

    # --- Save Results in same format as logic_phase_transition.png ---
    df = pd.DataFrame({
        "Step": history["epoch"],
        "Perfect_Bit_Identity_Rate": history["identity_rate"],
        "Bit_Weight_Kinetic_Energy_Kb": history["kb"]
    })
    
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/pp_emergence_log.csv", index=False)
    
    # --- Save Best Weights ---
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "pp_gpt_small_final.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    
    print(f"\nReal-Language PP Emergence Data saved to results/pp_emergence_log.csv")
    print(f"Model weights saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq_len", type=int, default=64)
    args = parser.parse_args()
    train_pp_emergence(args)
