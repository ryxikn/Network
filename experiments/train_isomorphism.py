import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

class LSTM_Cloner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM_Cloner, self).__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, c):
        h, c = self.lstm(x, (h, c))
        out = self.fc(h)
        return self.sigmoid(out), h, c

class Transformer_Cloner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=4, num_layers=2):
        super(Transformer_Cloner, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.embedding(x).unsqueeze(1) # (batch, 1, hidden_dim)
        x = self.transformer(x)
        x = x.squeeze(1)
        out = self.fc(x)
        return self.sigmoid(out)

def train_isomorphism(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    data = np.load(args.data_path)
    X_data = torch.tensor(data['inputs'], dtype=torch.float32).to(device)
    Y_data = torch.tensor(data['outputs'], dtype=torch.float32).to(device)
    
    input_dim = X_data.shape[1]
    output_dim = Y_data.shape[1]
    hidden_dim = args.hidden_dim
    
    # Initialize Models
    models = {}
    if args.model == 'all' or args.model == 'lstm':
        models['lstm'] = LSTM_Cloner(input_dim, hidden_dim, output_dim).to(device)
    if args.model == 'all' or args.model == 'transformer':
        models['transformer'] = Transformer_Cloner(input_dim, hidden_dim, output_dim).to(device)
    
    optimizers = {name: optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4) for name, m in models.items()}
    schedulers = {name: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) for name, opt in optimizers.items()}
    criterion = nn.BCELoss()
    
    history = {"epoch": []}
    for name in models:
        history[f"{name}_loss"] = []
        history[f"{name}_acc"] = []
        history[f"{name}_perfect"] = []
        history[f"{name}_kb"] = [] # Kinetic Energy

    print(f"Starting Isomorphism Training | Task: {args.data_path} | Models: {list(models.keys())} | Device: {device}")
    
    # Pre-train weight snapshots
    for name, model in models.items():
        torch.save(model.state_dict(), f"checkpoints/{name}_cloner_step_0.pth")

    prev_weights = {name: {k: v.clone() for k, v in m.state_dict().items()} for name, m in models.items()}

    for epoch in range(args.epochs):
        for m in models.values(): m.train()
        epoch_losses = {name: 0 for name in models}
        epoch_kb = {name: 0 for name in models}
        
        perm = torch.randperm(X_data.size(0) - args.seq_len)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i+args.batch_size]
            
            for name, model in models.items():
                optimizers[name].zero_grad()
                loss_seq = 0
                
                if name == 'lstm':
                    h = torch.zeros(len(idx), hidden_dim).to(device)
                    c = torch.zeros(len(idx), hidden_dim).to(device)
                    for s in range(args.seq_len):
                        out, h, c = model(X_data[idx + s], h, c)
                        loss_seq += criterion(out, Y_data[idx + s])
                else: # transformer
                    for s in range(args.seq_len):
                        out = model(X_data[idx + s])
                        loss_seq += criterion(out, Y_data[idx + s])
                
                loss_seq /= args.seq_len
                loss_seq.backward()
                optimizers[name].step()
                epoch_losses[name] += loss_seq.item()

        # Calculate Kinetic Energy (Kb) and save snapshots
        for name, model in models.items():
            kb = 0
            curr_weights = model.state_dict()
            for k in curr_weights:
                kb += torch.sum((curr_weights[k] - prev_weights[name][k])**2).item()
            epoch_kb[name] = kb
            prev_weights[name] = {k: v.clone() for k, v in curr_weights.items()}
            
            if epoch == 44: # Phase transition snapshot
                torch.save(model.state_dict(), f"checkpoints/{name}_cloner_step_45.pth")
        
        # Evaluation
        for m in models.values(): m.eval()
        with torch.no_grad():
            eval_metrics = {}
            for name, model in models.items():
                if name == 'lstm':
                    h0 = torch.zeros(X_data.size(0), hidden_dim).to(device)
                    c0 = torch.zeros(X_data.size(0), hidden_dim).to(device)
                    out, _, _ = model(X_data, h0, c0)
                else:
                    out = model(X_data)
                
                acc = ((out > 0.5) == (Y_data > 0.5)).float().mean().item()
                perfect = ((out > 0.5) == (Y_data > 0.5)).all(dim=1).float().mean().item()
                eval_metrics[f"{name}_acc"] = acc
                eval_metrics[f"{name}_perfect"] = perfect
                
        history["epoch"].append(epoch)
        for name in models:
            avg_loss = epoch_losses[name] / (len(perm) / args.batch_size)
            history[f"{name}_loss"].append(avg_loss)
            history[f"{name}_acc"].append(eval_metrics[f"{name}_acc"])
            history[f"{name}_perfect"].append(eval_metrics[f"{name}_perfect"])
            history[f"{name}_kb"].append(epoch_kb[name])
            schedulers[name].step()

        if (epoch + 1) % 20 == 0 or epoch == 44:
            log_str = f"Epoch {epoch+1:3d}"
            for name in models:
                log_str += f" | {name.upper()} Loss: {history[f'{name}_loss'][-1]:.6f} Acc: {history[f'{name}_acc'][-1]:.4f} Perf: {history[f'{name}_perfect'][-1]:.4f} Kb: {history[f'{name}_kb'][-1]:.4e}"
            print(log_str)

    # Save results
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    pd.DataFrame(history).to_csv("results/isomorphism_training_log.csv", index=False)
    for name, model in models.items():
        torch.save(model.state_dict(), f"checkpoints/{name}_cloner_final.pth")
    
    print("\nTraining Complete. Models and logs saved in checkpoints/ and results/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HLI Logic Isomorphism Training Script")
    parser.add_argument("--data_path", type=str, default="results/cpu_behavior_data.npz")
    parser.add_argument("--model", type=str, default="all", choices=['lstm', 'transformer', 'all'])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq_len", type=int, default=5)
    
    args = parser.parse_args()
    train_isomorphism(args)
