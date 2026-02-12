import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cpu"
    diffusion_steps: int = 64
    epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 256

    c: float = 1.0
    alpha: float = 1.0
    dt: float = 0.05
    u_max: float = 2.0
    state_bound: float = 2.5

    train_safe_samples: int = 16000
    calib_samples: int = 4000
    test_samples: int = 4000

    rollout_batch: int = 256
    rollout_horizon: int = 80


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DenoiserMLP(nn.Module):
    def __init__(self, state_dim: int = 2, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.time_net = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_net(t)
        return self.net(torch.cat([x_t, t_emb], dim=-1))


class DiffusionSchedule:
    def __init__(self, num_steps: int, device: torch.device, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bar[t].unsqueeze(-1)
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def true_cbf(x: torch.Tensor, c: float) -> torch.Tensor:
    # h(x) >= 0 is safe: inside an ellipsoidal set centered at origin.
    return c - 0.5 * torch.sum(x * x, dim=-1)


def dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    x1_dot = x[:, 1]
    x2_dot = u[:, 0]
    return torch.stack([x1_dot, x2_dot], dim=-1)


def euler_step(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    return x + dt * dynamics(x, u)


def sample_states(num_samples: int, bound: float, device: torch.device) -> torch.Tensor:
    return (2.0 * torch.rand(num_samples, 2, device=device) - 1.0) * bound


def sample_safe_states(num_samples: int, bound: float, c: float, device: torch.device) -> torch.Tensor:
    chunks = []
    collected = 0
    batch = max(1024, num_samples)
    while collected < num_samples:
        x = sample_states(batch, bound, device)
        mask = true_cbf(x, c) >= 0.0
        if mask.any():
            keep = x[mask]
            chunks.append(keep)
            collected += keep.shape[0]
    return torch.cat(chunks, dim=0)[:num_samples]


def project_control_with_barrier(
    x: torch.Tensor,
    u_nom: torch.Tensor,
    h_val: torch.Tensor,
    grad_h: torch.Tensor,
    alpha: float,
    u_max: float,
    eps: float = 1e-6,
):
    # Constraint: grad_h(x) * (f(x) + g(x) u) + alpha h(x) >= 0
    # For double integrator: f=[x2,0], g=[0,1].
    f = torch.stack([x[:, 1], torch.zeros_like(x[:, 0])], dim=-1)
    a = grad_h[:, 1]
    b = -alpha * h_val - torch.sum(grad_h * f, dim=-1)

    lower = torch.full_like(b, -float("inf"))
    upper = torch.full_like(b, float("inf"))

    pos = a > eps
    neg = a < -eps

    lower[pos] = b[pos] / a[pos]
    upper[neg] = b[neg] / a[neg]

    u = u_nom[:, 0]
    u = torch.maximum(u, lower)
    u = torch.minimum(u, upper)
    u = torch.clamp(u, -u_max, u_max)

    residual = a * u - b
    near_zero = torch.abs(a) <= eps
    feasible = residual >= -1e-5
    feasible = torch.where(near_zero, b <= 1e-5, feasible)

    return u.unsqueeze(-1), feasible


def train_diffusion(
    model: nn.Module,
    schedule: DiffusionSchedule,
    safe_states: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
):
    model.train()
    loader = DataLoader(TensorDataset(safe_states), batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0.0
        count = 0
        for (x0,) in loader:
            t = torch.randint(0, schedule.num_steps, (x0.shape[0],), device=x0.device)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)

            noise_hat = model(x_t, t)
            loss = torch.mean((noise_hat - noise) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item() * x0.shape[0]
            count += x0.shape[0]

        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:04d} diffusion_mse={total / max(count, 1):.6f}")


class LearnedDiffusionBarrier:
    def __init__(self, model: nn.Module, schedule: DiffusionSchedule, threshold: float, eval_timesteps):
        self.model = model
        self.schedule = schedule
        self.threshold = threshold
        self.eval_timesteps = eval_timesteps

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        # Deterministic denoising energy proxy: low inside learned safe-state manifold.
        e = torch.zeros(x.shape[0], device=x.device)
        for step in self.eval_timesteps:
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            scale = self.schedule.alpha_bar[step].sqrt()
            x_t = scale * x
            noise_hat = self.model(x_t, t)
            e = e + torch.mean(noise_hat * noise_hat, dim=-1)
        return e / len(self.eval_timesteps)

    def h(self, x: torch.Tensor) -> torch.Tensor:
        return self.threshold - self.energy(x)


def find_threshold(energy: torch.Tensor, labels_safe: torch.Tensor) -> float:
    # Choose threshold maximizing balanced accuracy on calibration data.
    qs = torch.linspace(0.01, 0.99, 99, device=energy.device)
    candidates = torch.quantile(energy, qs)

    best_thr = candidates[0].item()
    best_score = -1.0

    labels = labels_safe.bool()
    pos = labels
    neg = ~labels

    for thr in candidates:
        pred = energy <= thr
        tpr = (pred[pos].float().mean().item()) if pos.any() else 0.0
        tnr = ((~pred[neg]).float().mean().item()) if neg.any() else 0.0
        score = 0.5 * (tpr + tnr)
        if score > best_score:
            best_score = score
            best_thr = thr.item()

    return best_thr


def classification_metrics(pred_safe: torch.Tensor, true_safe: torch.Tensor):
    pred = pred_safe.bool()
    true = true_safe.bool()

    tp = torch.sum(pred & true).item()
    tn = torch.sum((~pred) & (~true)).item()
    fp = torch.sum(pred & (~true)).item()
    fn = torch.sum((~pred) & true).item()

    total = max(tp + tn + fp + fn, 1)
    acc = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def barrier_value_and_grad(barrier: LearnedDiffusionBarrier, x: torch.Tensor):
    x_req = x.detach().clone().requires_grad_(True)
    h = barrier.h(x_req)
    grad = torch.autograd.grad(h.sum(), x_req, create_graph=False)[0]
    return h.detach(), grad.detach()


def evaluate_closed_loop(
    barrier: LearnedDiffusionBarrier,
    c: float,
    alpha: float,
    dt: float,
    u_max: float,
    batch_size: int,
    horizon: int,
    state_bound: float,
    device: torch.device,
):
    x = sample_safe_states(batch_size, state_bound, c, device)

    violations = 0
    infeasible = 0
    total = 0

    for _ in range(horizon):
        u_nom = (2.0 * torch.rand(batch_size, 1, device=device) - 1.0) * u_max
        h_val, grad_h = barrier_value_and_grad(barrier, x)
        u, feasible = project_control_with_barrier(x, u_nom, h_val, grad_h, alpha, u_max)

        x = euler_step(x, u, dt)

        true_h_next = true_cbf(x, c)
        violations += torch.sum(true_h_next < 0.0).item()
        infeasible += torch.sum(~feasible).item()
        total += batch_size

    return {
        "unsafe_ratio": violations / max(total, 1),
        "infeasible_ratio": infeasible / max(total, 1),
    }


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    safe_train = sample_safe_states(cfg.train_safe_samples, cfg.state_bound, cfg.c, device)

    model = DenoiserMLP(state_dim=2, hidden_dim=cfg.hidden_dim).to(device)
    schedule = DiffusionSchedule(cfg.diffusion_steps, device=device)

    train_diffusion(
        model=model,
        schedule=schedule,
        safe_states=safe_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
    )

    model.eval()

    eval_ts = [0, cfg.diffusion_steps // 4, cfg.diffusion_steps // 2, 3 * cfg.diffusion_steps // 4, cfg.diffusion_steps - 1]

    with torch.no_grad():
        calib_x = sample_states(cfg.calib_samples, cfg.state_bound, device)
        calib_safe = true_cbf(calib_x, cfg.c) >= 0.0

        tmp_barrier = LearnedDiffusionBarrier(model, schedule, threshold=0.0, eval_timesteps=eval_ts)
        calib_energy = tmp_barrier.energy(calib_x)
        threshold = find_threshold(calib_energy, calib_safe)

        barrier = LearnedDiffusionBarrier(model, schedule, threshold=threshold, eval_timesteps=eval_ts)

        test_x = sample_states(cfg.test_samples, cfg.state_bound, device)
        test_safe = true_cbf(test_x, cfg.c) >= 0.0
        pred_safe = barrier.h(test_x) >= 0.0
        metrics = classification_metrics(pred_safe, test_safe)

    print("\n=== Learned Diffusion Barrier Metrics ===")
    print(f"threshold={threshold:.6f}")
    print(f"accuracy={metrics['acc']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
    print(f"confusion tp={metrics['tp']} tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']}")

    closed_loop = evaluate_closed_loop(
        barrier=barrier,
        c=cfg.c,
        alpha=cfg.alpha,
        dt=cfg.dt,
        u_max=cfg.u_max,
        batch_size=cfg.rollout_batch,
        horizon=cfg.rollout_horizon,
        state_bound=cfg.state_bound,
        device=device,
    )

    print("\n=== Closed-Loop Safety Check (using learned barrier in safety filter) ===")
    print(f"unsafe_state_ratio={closed_loop['unsafe_ratio']:.6f}")
    print(f"projection_infeasible_ratio={closed_loop['infeasible_ratio']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion-based Control Barrier Function surrogate on a double-integrator system.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--u-max", type=float, default=2.0)
    parser.add_argument("--state-bound", type=float, default=2.5)

    parser.add_argument("--train-safe-samples", type=int, default=16000)
    parser.add_argument("--calib-samples", type=int, default=4000)
    parser.add_argument("--test-samples", type=int, default=4000)

    parser.add_argument("--rollout-batch", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=80)

    args = parser.parse_args()
    main(TrainConfig(**vars(args)))
