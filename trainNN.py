import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def control_barrier_function(x: torch.Tensor, c: float) -> torch.Tensor:
    # Matches the legacy MATLAB sign convention used in this repo.
    return 0.5 * torch.sum(x * x, dim=-1, keepdim=True) - c


class CBFRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class NormalizedCBFModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = self.base_model(x_norm)
        return y_norm * self.y_std + self.y_mean


def load_states_from_pt(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "safe_trajectories" not in payload:
        raise ValueError(f"{path} must contain a dict with key 'safe_trajectories'.")
    safe_trajectories = payload["safe_trajectories"].float()
    if safe_trajectories.ndim != 2 or safe_trajectories.shape[0] != 2:
        raise ValueError(f"safe_trajectories must have shape [2, N], got {tuple(safe_trajectories.shape)}.")
    return safe_trajectories.T.contiguous()  # [N, 2]


def normalize(x: torch.Tensor):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std, mean, std


def maybe_export_onnx(
    model: nn.Module,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    output_path: str,
):
    wrapped = NormalizedCBFModel(model, x_mean, x_std, y_mean, y_std).eval()
    dummy_input = torch.randn(1, 2, dtype=torch.float32)
    try:
        torch.onnx.export(
            wrapped,
            dummy_input,
            output_path,
            input_names=["state"],
            output_names=["h"],
            dynamic_axes={"state": {0: "batch_size"}, "h": {0: "batch_size"}},
        )
        print(f"Exported ONNX model to {output_path}")
    except Exception as exc:
        print(f"ONNX export skipped: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Train a CBF regressor in pure Python (no MATLAB required).")
    parser.add_argument("--data", type=str, default="safe_trajectories.pt", help="Input dataset from trainingDataGen.py.")
    parser.add_argument("--c", type=float, default=1.0, help="Safe set constant used for target CBF labels.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-out", type=str, default="cbf_model.pth")
    parser.add_argument("--onnx-out", type=str, default="cbf_model.onnx")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export.")
    args = parser.parse_args()

    X = load_states_from_pt(args.data)
    y = control_barrier_function(X, args.c)

    X_norm, X_mean, X_std = normalize(X)
    y_norm, y_mean, y_std = normalize(y)

    train_loader = DataLoader(
        TensorDataset(X_norm, y_norm),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = CBFRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_count = 0
        model.train()

        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg = total_loss / max(total_count, 1)
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {avg:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": X_mean,
        "input_std": X_std,
        "output_mean": y_mean,
        "output_std": y_std,
        "c": args.c,
    }
    torch.save(checkpoint, args.model_out)
    print(f"Saved trained model to {args.model_out}")

    if not args.skip_onnx:
        maybe_export_onnx(model, X_mean, X_std, y_mean, y_std, args.onnx_out)


if __name__ == "__main__":
    main()
