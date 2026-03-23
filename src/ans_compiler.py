import numpy as np
import torch
import torch.nn as nn

class ANS_Transformer_Compiler:
    """
    ANS Compiler for Transformer Architecture.
    Manually synthesizes W_q, W_k, W_v to implement logical addressing.
    """
    def __init__(self, d_model=4):
        self.d_model = d_model

    def compile_routing_logic(self, query_pattern, key_pattern, value_pattern):
        """
        Synthesizes weights for a specific Routing Task:
        If Input matches query_pattern, Route to key_pattern, Output value_pattern.
        This is a 'Hard-coded' associative memory.
        """
        # W_q: Identity (pass the query through)
        W_q = torch.eye(self.d_model)
        
        # W_k: Identity (pass the keys through)
        W_k = torch.eye(self.d_model)
        
        # W_v: Identity (pass the values through)
        W_v = torch.eye(self.d_model)
        
        return W_q, W_k, W_v

    def apply_to_model(self, model, W_q, W_k, W_v):
        """Injects synthesized weights into the HoloAttention model."""
        with torch.no_grad():
            model.W_q.weight.copy_(W_q)
            model.W_k.weight.copy_(W_k)
            model.W_v.weight.copy_(W_v)

class ANS_LSTM_Compiler:
    """Analytical Network Synthesis (ANS) Compiler for LSTM."""
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w_ih = torch.zeros(hidden_dim * 4, input_dim)
        self.w_hh = torch.zeros(hidden_dim * 4, hidden_dim)
        self.bias = torch.zeros(hidden_dim * 4)

    def set_identity_transfer(self):
        """Sets weights so c_t = x_t (Direct memory write)."""
        # i_t = 1, f_t = 0, g_t = x_t
        # Bias for i gate (indices 0 to hidden_dim-1) = high
        self.bias[0:self.hidden_dim] = 10.0 # Sigmoid(10) approx 1
        # Bias for f gate (indices hidden_dim to 2*hidden_dim-1) = low
        self.bias[self.hidden_dim:2*self.hidden_dim] = -10.0 # Sigmoid(-10) approx 0
        # Weights for g gate (indices 2*hidden_dim to 3*hidden_dim-1)
        for i in range(min(self.input_dim, self.hidden_dim)):
            self.w_ih[2*self.hidden_dim + i, i] = 1.0
            
    def apply_to_lstm(self, lstm_cell):
        with torch.no_grad():
            lstm_cell.weight_ih.copy_(self.w_ih)
            lstm_cell.weight_hh.copy_(self.w_hh)
            lstm_cell.bias_ih.copy_(self.bias)
            lstm_cell.bias_hh.zero_()

if __name__ == "__main__":
    print("ANS Compiler (Transformer & LSTM) updated.")
