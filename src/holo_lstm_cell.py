import numpy as np
from holo_logic_gates import HoloLogicGates

class HoloLSTMCell:
    """
    Implementation of Theorem 3: Analytical Isomorphism of LSTM.
    Reconstructs the LSTM Gating mechanism as a composition of Holomorphic Operators.
    """
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gates = HoloLogicGates()
        
        # Initialize weights (Complex-ready dtype)
        self.W_i = np.random.randn(hidden_dim, input_dim + hidden_dim).astype(complex)
        self.W_f = np.random.randn(hidden_dim, input_dim + hidden_dim).astype(complex)
        self.W_o = np.random.randn(hidden_dim, input_dim + hidden_dim).astype(complex)
        self.W_c = np.random.randn(hidden_dim, input_dim + hidden_dim).astype(complex)

    def forward(self, x, h_prev, c_prev):
        """
        Analytical Forward Pass using Gating Operators.
        Note: We use soft-logic (Holo-Gates) instead of Sigmoid/Tanh
        to maintain pure polynomial/holomorphic property.
        """
        # Concatenate input and previous hidden state
        concat = np.concatenate([x, h_prev], axis=0)
        
        # 1. Gate Pre-activations (Linear Fields)
        # In standard LSTM, these go through Sigmoid.
        # Here, they define the "Logical Context" z.
        i_field = np.dot(self.W_i, concat)
        f_field = np.dot(self.W_f, concat)
        o_field = np.dot(self.W_o, concat)
        c_tilde_field = np.dot(self.W_c, concat)
        
        # 2. Holomorphic Gating (Lemma 1 & 2)
        # We treat the fields as analytical signals. 
        # For pure logic verification, fields would be in {0, 1}.
        # For general neural dynamics, they are in C.
        
        # Memory Update: c_t = f_t * c_{t-1} + i_t * c_tilde_t
        # This is exactly the AND-OR structure in your derivation.
        term_forget = self.gates.AND(f_field, c_prev)
        term_input  = self.gates.AND(i_field, c_tilde_field)
        
        # Analytical Cell State (Polynomial Composition)
        c_t = self.gates.OR(term_forget, term_input)
        
        # 3. Output Gating
        h_t = self.gates.AND(o_field, c_t)
        
        return h_t, c_t

if __name__ == "__main__":
    # Test logic isomorphism: Can it act as a Flip-Flop / Counter?
    cell = HoloLSTMCell(input_dim=1, hidden_dim=1)
    
    # Manually set weights to simulate a "Pass-Through" logic
    cell.W_i[:] = 0; cell.W_i[0, 0] = 1.0 # Input gate always active if x=1
    cell.W_f[:] = 0; cell.W_f[0, 1] = 1.0 # Forget gate active if h_prev=1
    
    h = np.array([0.0])
    c = np.array([0.0])
    
    print("--- Holo-LSTM: Recursive Logic Test ---")
    for t in range(3):
        x = np.array([1.0]) # Keep feeding 1
        h, c = cell.forward(x, h, c)
        print(f"Step {t}: h={h[0]}, c={c[0]}")
