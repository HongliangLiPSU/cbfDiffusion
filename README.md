## CBF Diffusion

Python-only workflows for learning and validating Control Barrier Functions (CBFs) on a double-integrator system.

### Project Layout

```text
.
├── artifacts/
│   ├── data/          # Generated trajectory files (.pt)
│   └── models/        # Trained models (.pth/.onnx)
├── legacy/
│   ├── experiments/   # Older exploratory scripts
│   ├── matlab/        # Original MATLAB scripts
│   └── data/          # Legacy .mat files
├── diffusion_cbf.py   # Diffusion-based CBF training/evaluation
├── trainingDataGen.py # Generate safe trajectories (Python replacement)
├── trainNN.py         # Train neural CBF regressor
├── validateNN.py      # Validate learned CBF in closed-loop simulation
└── Makefile
```

### Setup

```bash
python3 -m pip install -r requirements.txt
```

### Standard CBF Workflow

```bash
python3 trainingDataGen.py
python3 trainNN.py
python3 validateNN.py
```

Equivalent `make` targets:

```bash
make generate-data
make train-cbf
make validate-cbf
```

### Diffusion-Based CBF Workflow

```bash
python3 diffusion_cbf.py --epochs 300 --device cpu
```

This prints:

- diffusion training loss,
- safe/unsafe classification metrics (`accuracy`, `precision`, `recall`, `f1`),
- closed-loop safety metrics (`unsafe_state_ratio`, `projection_infeasible_ratio`).
