## Learning Control Barrier Functions (Python-Only)

This repo can now run end-to-end without MATLAB.

### Python Replacement for Legacy MATLAB Flow

1. Generate trajectories:

```bash
python3 trainingDataGen.py --output safe_trajectories.pt
```

2. Train CBF regressor:

```bash
python3 trainNN.py --data safe_trajectories.pt --model-out cbf_model.pth
```

3. Validate in simulation:

```bash
python3 validateNN.py --model cbf_model.pth --output safe_trajectories_learned.pt
```

This replaces:

- `trainingDataGen.m` -> `trainingDataGen.py`
- `validateNN.m` -> `validateNN.py`

### Diffusion-Based CBF Workflow

For the diffusion-model CBF pipeline on the double integrator:

```bash
python3 diffusion_cbf.py --epochs 300 --device cpu
```

It prints:

- diffusion training loss,
- safe/unsafe classification metrics (`accuracy`, `precision`, `recall`, `f1`),
- closed-loop safety metrics (`unsafe_state_ratio`, `projection_infeasible_ratio`).
