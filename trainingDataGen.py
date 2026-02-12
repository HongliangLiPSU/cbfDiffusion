import argparse

import torch


def double_integrator(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return torch.stack([x[..., 1], u[..., 0]], dim=-1)


def control_barrier_function(x: torch.Tensor, c: float) -> torch.Tensor:
    # Matches the legacy MATLAB sign convention.
    return 0.5 * torch.sum(x * x, dim=-1) - c


def simulate(c: float, x0, T: float, dt: float):
    steps = int(round(T / dt))
    t = torch.linspace(0.0, T, steps + 1, dtype=torch.float32)

    x = torch.zeros(steps + 1, 2, dtype=torch.float32)
    u = torch.zeros(steps, 1, dtype=torch.float32)
    x[0] = torch.tensor(x0, dtype=torch.float32)

    for k in range(steps):
        u_k = -x[k, 0] - x[k, 1]
        if control_barrier_function(x[k : k + 1], c).item() < 0.0:
            u_k = -0.1 * x[k, 0] - 0.1 * x[k, 1]

        u[k, 0] = u_k
        dx = double_integrator(x[k : k + 1], u[k : k + 1]).squeeze(0)
        x[k + 1] = x[k] + dt * dx

    safe_trajectories = x.T.contiguous()  # Shape: [2, num_steps + 1]
    return t, x, u, safe_trajectories


def maybe_plot(t: torch.Tensor, x: torch.Tensor):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    t_list = t.tolist()
    x1 = x[:, 0].tolist()
    x2 = x[:, 1].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t_list, x1, color="r", linewidth=1.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position")
    axes[0].set_title("Double Integrator System with CBF")

    axes[1].plot(t_list, x2, color="b", linewidth=1.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity")

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate safe trajectories for the double-integrator CBF example.")
    parser.add_argument("--c", type=float, default=1.0, help="Safe set constant.")
    parser.add_argument("--x0", type=float, nargs=2, default=[1.0, 0.0], help="Initial state [x1, x2].")
    parser.add_argument("--T", type=float, default=100.0, help="Simulation time horizon.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step.")
    parser.add_argument(
        "--output",
        type=str,
        default="safe_trajectories.pt",
        help="Output file (.pt) containing safe_trajectories and rollout data.",
    )
    parser.add_argument("--plot", action="store_true", help="Plot position and velocity trajectories.")
    args = parser.parse_args()

    t, x, u, safe_trajectories = simulate(c=args.c, x0=args.x0, T=args.T, dt=args.dt)

    payload = {
        "safe_trajectories": safe_trajectories,
        "time": t,
        "states": x,
        "controls": u,
        "c": args.c,
        "dt": args.dt,
    }
    torch.save(payload, args.output)

    print(f"Saved safe trajectories to {args.output}")
    print(f"safe_trajectories shape: {tuple(safe_trajectories.shape)}")

    if args.plot:
        maybe_plot(t, x)


if __name__ == "__main__":
    main()
