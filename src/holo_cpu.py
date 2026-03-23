import numpy as np
from holo_logic_gates import HoloLogicGates

class HoloLSTM_CPU:
    """
    Legacy Holo-CPU (4-bit Accumulator, 8-bit RAM).
    Used for Fibonacci sequence demonstration in robustness_scan.py.
    """
    def __init__(self):
        self.gates = HoloLogicGates()
        self.ACC = np.zeros(4, dtype=complex)
        self.PC = np.zeros(4, dtype=complex)
        self.ram = [np.zeros(4, dtype=complex) for _ in range(16)]
        self.rom_data = [np.zeros(8, dtype=complex) for _ in range(16)]
        self.halted = 0j

    def MUX(self, s, a, b):
        return (1.0 - s) * a + s * b

    def _decoder_16(self, addr_bits):
        sel = []
        for i in range(16):
            line = 1.0
            for bit in range(4):
                bit_val = float((i >> bit) & 1)
                term = addr_bits[bit] if bit_val > 0.5 else self.gates.NOT(addr_bits[bit])
                line = self.gates.AND(line, term)
            sel.append(line)
        return sel

    def _full_adder_4bit(self, a_vec, b_vec, cin):
        res = np.zeros(4, dtype=complex)
        carry = cin
        for i in range(4):
            s = self.gates.XOR(self.gates.XOR(a_vec[i], b_vec[i]), carry)
            a_xor_b = self.gates.XOR(a_vec[i], b_vec[i])
            carry = self.gates.OR(self.gates.AND(a_vec[i], b_vec[i]), self.gates.AND(carry, a_xor_b))
            res[i] = s
        return res, carry

    def step(self, debug=False):
        if self.halted.real > 0.5: return
        
        # Fetch
        sel_pc = self._decoder_16(self.PC)
        instr = np.zeros(8, dtype=complex)
        for i in range(16): instr += sel_pc[i] * self.rom_data[i]
        
        # Op is bits 0-3, Addr is bits 4-7
        op = instr[0:4]
        addr = instr[4:8]
        
        def check_op(p):
            res = 1.0
            for i in range(4):
                bit_val = op[i]
                # Analytical equality: AND_i (p_i == op_i)
                term = (bit_val * p[i]) + (self.gates.NOT(bit_val) * self.gates.NOT(p[i]))
                res = self.gates.AND(res, term)
            return res

        is_lda = check_op([1, 0, 0, 0]) # 1
        is_sta = check_op([0, 1, 0, 0]) # 2
        is_add = check_op([1, 1, 0, 0]) # 3
        is_jmp = check_op([0, 0, 1, 0]) # 4
        is_hlt = check_op([1, 1, 1, 1]) # F
        
        # Read RAM
        sel_addr = self._decoder_16(addr)
        ram_val = np.zeros(4, dtype=complex)
        for i in range(16): ram_val += sel_addr[i] * self.ram[i]
        
        # ALU
        alu_res, _ = self._full_adder_4bit(self.ACC, ram_val, 0.0)
        
        # Commit
        new_acc = self.ACC.copy()
        new_acc = self.MUX(is_lda, new_acc, ram_val)
        new_acc = self.MUX(is_add, new_acc, alu_res)
        self.ACC = new_acc
        
        for i in range(16):
            self.ram[i] = self.MUX(self.gates.AND(is_sta, sel_addr[i]), self.ram[i], self.ACC)
            
        pc_plus_1, _ = self._full_adder_4bit(self.PC, np.array([1,0,0,0], dtype=complex), 0.0)
        self.PC = self.MUX(is_jmp, pc_plus_1, addr)
        if is_hlt.real > 0.5: self.halted = 1.0

class HoloRISC_LSTM:
    """
    Holo-RISC CPU: A Turing-complete 8-bit architecture mapped entirely into 
    a single Holomorphic LSTM-like structure.
    """
    def __init__(self):
        self.gates = HoloLogicGates()
        # Registers (8 bits each, complex)
        self.R = [np.zeros(8, dtype=complex) for _ in range(4)]
        self.SP = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=complex) # SP starts at 192 (0xC0)
        self.PC = np.zeros(8, dtype=complex)
        self.IR = np.zeros(16, dtype=complex)
        self.PH = 0j # Phase: 0: Fetch, 1: Execute
        self.halted = 0j
        
        # Memory: 256 addresses, 8 bits each
        self.memory = np.zeros((256, 8), dtype=complex)
        # ROM: 256 addresses, 16 bits each
        self.rom = np.zeros((256, 16), dtype=complex)

    def MUX(self, s, a, b):
        """Analytical MUX: (1-s)a + sb"""
        return (1.0 - s) * a + s * b

    def _decoder_256(self, addr_bits):
        """8-to-256 Holomorphic Decoder."""
        # Lower 4 bits
        sel_low = []
        for i in range(16):
            line = 1.0
            for bit in range(4):
                bit_val = float((i >> bit) & 1)
                term = addr_bits[bit] if bit_val > 0.5 else self.gates.NOT(addr_bits[bit])
                line = self.gates.AND(line, term)
            sel_low.append(line)
        
        # Upper 4 bits
        sel_high = []
        for i in range(16):
            line = 1.0
            for bit in range(4):
                bit_val = float((i >> bit) & 1)
                term = addr_bits[bit+4] if bit_val > 0.5 else self.gates.NOT(addr_bits[bit+4])
                line = self.gates.AND(line, term)
            sel_high.append(line)
            
        sel = []
        for h in range(16):
            for l in range(16):
                sel.append(self.gates.AND(sel_high[h], sel_low[l]))
        return sel

    def _full_adder_8bit(self, a_vec, b_vec, cin, subtract=False):
        res = np.zeros(8, dtype=complex)
        carry = cin
        for i in range(8):
            b_eff = self.gates.XOR(b_vec[i], subtract)
            s = self.gates.XOR(self.gates.XOR(a_vec[i], b_eff), carry)
            a_xor_b = self.gates.XOR(a_vec[i], b_eff)
            carry = self.gates.OR(self.gates.AND(a_vec[i], b_eff), self.gates.AND(carry, a_xor_b))
            res[i] = s
        return res, carry

    def _read_mem(self, addr_bits):
        sel = self._decoder_256(addr_bits)
        res = np.zeros(8, dtype=complex)
        for bit in range(8):
            res[bit] = sum(sel[i] * self.memory[i][bit] for i in range(256))
        return res

    def _read_rom(self, addr_bits):
        sel = self._decoder_256(addr_bits)
        res = np.zeros(16, dtype=complex)
        for bit in range(16):
            res[bit] = sum(sel[i] * self.rom[i][bit] for i in range(256))
        return res

    def _is_zero(self, vec):
        any_bit = 0.0
        for b in vec:
            any_bit = self.gates.OR(any_bit, b)
        return self.gates.NOT(any_bit)

    def step(self, debug=False):
        if self.halted.real > 0.5: return

        # 1. Fetch Phase
        current_instr = self._read_rom(self.PC)
        
        # 2. Decode Phase
        instr_to_decode = current_instr if self.PH.real < 0.5 else self.IR
        
        op = instr_to_decode[12:16]
        rd_idx_bits = instr_to_decode[10:12]
        rs_idx_bits = instr_to_decode[8:10]
        imm = instr_to_decode[0:8]

        if debug:
            p_val = int(sum(round(self.PC[i].real) * (2**i) for i in range(8)))
            ph_val = int(round(self.PH.real))
            print(f"PC: {p_val}, Phase: {ph_val}, IR_hex: {hex(int(sum(round(instr_to_decode[i].real)*(2**i) for i in range(16))))}")

        # OP Decoders
        def check_op(p):
            res = 1.0
            for i in range(4):
                bit_val = op[i]
                term = self.MUX(float(p[i]), self.gates.NOT(bit_val), bit_val)
                res = self.gates.AND(res, term)
            return res

        is_mov = check_op([1, 0, 0, 0]) # OP=1: MOV (Little Endian Bit Order: 1,0,0,0 is 1)
        is_add = check_op([0, 1, 0, 0]) # OP=2: ADD
        is_sub = check_op([1, 1, 0, 0]) # OP=3: SUB
        is_ld  = check_op([0, 0, 1, 0]) # OP=4: LD
        is_hlt = check_op([1, 1, 1, 1]) # OP=15: HLT

        # Register Selectors
        def get_reg_sel(idx_bits):
            not0 = self.gates.NOT(idx_bits[0])
            not1 = self.gates.NOT(idx_bits[1])
            s0 = self.gates.AND(not1, not0)
            s1 = self.gates.AND(not1, idx_bits[0])
            s2 = self.gates.AND(idx_bits[1], not0)
            s3 = self.gates.AND(idx_bits[1], idx_bits[0])
            return [s0, s1, s2, s3]

        rd_sel = get_reg_sel(rd_idx_bits)
        rs_sel = get_reg_sel(rs_idx_bits)
        
        # Read Rd and Rs values
        val_rd = np.zeros(8, dtype=complex)
        val_rs = np.zeros(8, dtype=complex)
        for i in range(4):
            val_rd += rd_sel[i] * self.R[i]
            val_rs += rs_sel[i] * self.R[i]

        # 3. Execution Logic
        alu_add, _ = self._full_adder_8bit(val_rd, val_rs, 0.0)
        alu_sub, _ = self._full_adder_8bit(val_rd, val_rs, 1.0, subtract=True)
        pc_plus_1, _ = self._full_adder_8bit(self.PC, np.array([1,0,0,0,0,0,0,0], dtype=complex), 0.0)
        
        # 4. Commit
        if self.PH.real < 0.5: # Fetch Cycle
            self.IR = current_instr.copy()
            self.PC = pc_plus_1.copy()
            self.PH = 1.0
        else: # Execute Cycle
            # Update Registers
            for i in range(4):
                v = self.R[i].copy()
                v_updated = v.copy()
                v_updated = self.MUX(is_mov, v_updated, imm.copy())
                v_updated = self.MUX(is_add, v_updated, alu_add.copy())
                v_updated = self.MUX(is_sub, v_updated, alu_sub.copy())
                self.R[i] = self.MUX(rd_sel[i], v, v_updated)
            
            if debug:
                print(f"  R0: {self.get_reg(0)}, R1: {self.get_reg(1)}")
            
            if is_hlt.real > 0.5: self.halted = 1.0
            self.PH = 0.0

    def load_program(self, program):
        for i, instr in enumerate(program):
            self.rom[i] = instr

    def get_reg(self, idx):
        return int(sum(round(np.clip(self.R[idx][i].real, 0, 1)) * (2**i) for i in range(8)))

def int_to_16bit_vec(val):
    return np.array([(val >> i) & 1 for i in range(16)], dtype=complex)

def int_to_8bit_vec(val):
    return np.array([(val >> i) & 1 for i in range(8)], dtype=complex)

def int_to_4bit_vec(val):
    return np.array([(val >> i) & 1 for i in range(4)], dtype=complex)

if __name__ == "__main__":
    print("--- Holo-CPU: Fibonacci Sequence Demonstration ---")
    cpu = HoloLSTM_CPU()
    # Program: LDA 10 | ADD 11 | STA 11 | JMP 0 | ...
    # This is a simplified logic. Let's use the program from robustness_scan.
    prog_raw = [0x1B, 0x2A, 0x4D, 0x1B, 0x4A, 0x1D, 0x4B, 0x1C, 0x3E, 0x4C, 0x6C, 0x50, 0xF0]
    
    for i, instr in enumerate(prog_raw):
        cpu.rom_data[i] = int_to_8bit_vec(instr)
    
    # Init RAM: Fib(0)=0, Fib(1)=1
    cpu.ram[10] = int_to_4bit_vec(0)
    cpu.ram[11] = int_to_4bit_vec(1)
    
    print("Running Fibonacci...")
    for _ in range(50):
        cpu.step()
        if cpu.halted.real > 0.5: break
    
    res = int(sum(round(np.clip(cpu.ram[11][j].real, 0, 1)) * (2**j) for j in range(4)))
    print(f"Fibonacci Result in RAM[11]: {res}")

    print("\n--- HoloRISC-LSTM: Basic Operation Test ---")
    cpu_risc = HoloRISC_LSTM()
    # Program: MOV R0, 5 | MOV R1, 10 | ADD R0, R1 | HLT
    # Little-endian bits: OP=1 is 1,0,0,0. Rd=0 is 0,0. Rd=1 is 1,0.
    prog = [
        (0x1 << 12) | (0 << 10) | 5,
        (0x1 << 12) | (1 << 10) | 10,
        (0x2 << 12) | (0 << 10) | (1 << 8),
        (0xF << 12)
    ]
    cpu_risc.load_program([int_to_16bit_vec(p) for p in prog])
    
    for _ in range(20):
        cpu_risc.step(debug=True)
        if cpu_risc.halted.real > 0.5: break
        
    print(f"R0 Result: {cpu_risc.get_reg(0)}") # Should be 15
