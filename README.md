# Holomorphic Logic Isomorphism (HLI) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of the **Holomorphic Logic Isomorphism (HLI)** framework. HLI establishes a structural mapping between continuous neural architectures and discrete digital logic circuits through analytic continuation in the complex field $\mathbb{C}$.

## Core Concept

Traditional deep learning treats neural networks as statistical approximators. HLI redefines them as **Functional Realizations** of Boolean circuits. By mapping gating mechanisms to holomorphic polynomials, we prove that gated networks (LSTMs, Transformers) satisfy exact algebraic identities with hardware components such as MUXs, Registers, and ALUs.

## Repository Structure

The project is organized into a modular academic structure:

```text
.
├── src/                    # Core Analytical Implementation
│   ├── holo_logic_gates.py # Lemma 1: Holomorphic Polynomial Gates
│   ├── holo_lstm_cell.py   # Identity: Gated Recurrent Units as Registers
│   ├── holo_transformer.py # Identity: Self-Attention as Dynamic MUX/CAM
│   ├── ans_compiler.py    # Analytical Network Synthesis (ANS) Engine
│   ├── gpt_to_verilog.py   # Logic-to-Hardware Verilog Compiler (CSD mapping)
│   ├── gpt_small_full_logic.v # Synthesized 12-layer GPT-Small Netlist
│   └── pp_logic_scanner.py # Interpretability: Protocol Decoding for Language Tasks
├── experiments/            # Research & Validation Scripts
│   ├── train_isomorphism.py# Unified Training Engine (LSTM/Transformer)
│   ├── train_pp_emergence.py# High-Entropy Language Prediction (PP) Training
│   ├── large_model_logic_probe.py # Verification: 1.0000 EEF Scaling Law Scanner
│   ├── robustness_scan.py  # Lemma 2: Topological Stability & Quantization
│   ├── generate_pp_plots.py # Visualization: Figure 6 Logic Phase Transition
│   ├── generate_evolution_plots.py # Visualization: Figure 4 Logic Phase Transition
│   ├── generate_convergence_plots.py # Visualization: Figure 4 Convergence (Panels a, b, c)
│   └── har_causal_scanner.py # Interpretability: Sensor-to-Logic Mapping
├── checkpoints/            # Key Logic States (step_0, step_45, final)
├── results/                # 1.0000 Identity Results, Scaling Law Plots, and Logs
└── README.md               # Project Documentation
```

## Key Features

- **Analytical Network Synthesis (ANS)**: Direct compilation of optimal neural weights from Boolean specifications, bypassing stochastic optimization (SGD).
- **The Compilation Pipeline**: A 3-step synthesis flow (Analytic Extension $\to$ ISA Synthesis $\to$ Physical Mapping).
- **Bit-True Hardware Synthesis**: Compilers to transform trained weights into cycle-accurate Verilog netlists for FPGA deployment.
- **Topological Stability**: Mathematical proof of 16-bit absolute identity (1.0000) across models up to 1.5B parameters.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Transformers, NumPy, Pandas, Matplotlib

### Installation

```bash
git clone https://github.com/your-repo/holomorphic-logic-isomorphism.git
cd holomorphic-logic-isomorphism
pip install -r requirements.txt
```

### Running Experiments

To perform a 1.0000 EEF Scaling Law scan on pre-trained models:

```bash
python experiments/large_model_logic_probe.py
```

To synthesize a bit-true Verilog netlist from GPT-Small:

```bash
python src/gpt_to_verilog.py
```

To perform a robustness scan under quantization:

```bash
python experiments/robustness_scan.py --checkpoint checkpoints/transformer_cloner_final.pth
```

To regenerate the Figure 4 (Convergence) plot:

```bash
python experiments/generate_convergence_plots.py
```

## Mathematical Foundation

The HLI framework is built upon the **Boolean Ring** embedding:
- $\text{AND}(x, y) \equiv x \cdot y$
- $\text{OR}(x, y) \equiv x + y - x \cdot y$
- $\text{NOT}(x) \equiv 1 - x$

Under the mapping $\Phi$, neural components satisfy the identity:
$$ \Phi(c_t) = (1 - rst_t) \Phi(c_{t-1}) + we_t \Phi(d_{in}) $$

## License

This project is licensed under the MIT License - see the LICENSE file for details.
