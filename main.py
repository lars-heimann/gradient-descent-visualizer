"""Streamlit application to visualize gradient descent optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ScalarFunction = Callable[[float, float], float]
GradientFunction = Callable[[float, float], np.ndarray]
SchedulerParams = Dict[str, object]

OPTIMIZER_OPTIONS = [
    "Regular",
    "Momentum",
    "Nesterov",
    "AdaGrad",
    "RMSProp",
    "Adam",
    "Custom",
]

SCHEDULER_OPTIONS = [
    "None",
    "Power",
    "Exponential",
    "Piecewise",
    "Performance",
    "One Cycle",
]
EPSILON = 1e-8


@dataclass(frozen=True)
class ObjectiveFunction:
    """Container for an objective function and its metadata."""

    name: str
    description: str
    function: ScalarFunction
    gradient: GradientFunction
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]

    def surface(
        self, resolution: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a grid for Plotly visualizations."""

        x = np.linspace(self.x_range[0], self.x_range[1], resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], resolution)
        grid_x, grid_y = np.meshgrid(x, y)
        vectorized = np.vectorize(self.function)
        grid_z = vectorized(grid_x, grid_y)
        return grid_x, grid_y, grid_z


@dataclass
class OptimizerResult:
    """Stores the optimization trajectory of an optimizer run."""

    name: str
    positions: np.ndarray
    losses: List[float]


def steep_valley_function() -> ObjectiveFunction:
    def f(x: float, y: float) -> float:
        return 0.5 * x**2 + 2.0 * y**2

    def grad(x: float, y: float) -> np.ndarray:
        return np.array([x, 4.0 * y], dtype=float)

    return ObjectiveFunction(
        name="Steep Valley",
        description="Anisotropic quadratic bowl with different curvature per axis.",
        function=f,
        gradient=grad,
        x_range=(-20.0, 20.0),
        y_range=(-20.0, 20.0),
    )


def symmetric_convex_function() -> ObjectiveFunction:
    def f(x: float, y: float) -> float:
        return x**2 + y**2

    def grad(x: float, y: float) -> np.ndarray:
        return np.array([2.0 * x, 2.0 * y], dtype=float)

    return ObjectiveFunction(
        name="Symmetric Convex",
        description="Isotropic convex bowl – perfect baseline for comparing optimizers.",
        function=f,
        gradient=grad,
        x_range=(-15.0, 15.0),
        y_range=(-15.0, 15.0),
    )


def asymmetric_convex_function() -> ObjectiveFunction:
    def f(x: float, y: float) -> float:
        return (
            0.3 * (x - 10.0) ** 2
            + 0.2 * (y - 10.0) ** 2
            + 1.1 * (x + 10.0) ** 2
            + 1.0 * (y + 10.0) ** 2
        )

    def grad(x: float, y: float) -> np.ndarray:
        return np.array(
            [0.6 * (x - 10.0) + 2.2 * (x + 10.0), 0.4 * (y - 10.0) + 2.0 * (y + 10.0)]
        )

    return ObjectiveFunction(
        name="Asymmetric Convex",
        description="Convex bowl with shifted optimum and uneven curvature.",
        function=f,
        gradient=grad,
        x_range=(-20.0, 20.0),
        y_range=(-20.0, 20.0),
    )


def l_shaped_function() -> ObjectiveFunction:
    def f(x: float, y: float) -> float:
        return (y + 20.0) ** 2 + x if x > y else (x + 20.0) ** 2 + y

    def grad(x: float, y: float) -> np.ndarray:
        grad_x = 1.0 if x > y else 2.0 * (x + 20.0)
        grad_y = 2.0 * (y + 20.0) if x > y else 1.0
        return np.array([grad_x, grad_y], dtype=float)

    return ObjectiveFunction(
        name="L-shaped Valley",
        description="Piecewise valley that challenges momentum-based optimizers.",
        function=f,
        gradient=grad,
        x_range=(-50.0, 20.0),
        y_range=(-50.0, 20.0),
    )


def get_objectives() -> Dict[str, ObjectiveFunction]:
    return {
        obj.name: obj
        for obj in (
            steep_valley_function(),
            symmetric_convex_function(),
            asymmetric_convex_function(),
            l_shaped_function(),
        )
    }


def parse_float_sequence(raw_value: str) -> List[float]:
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def apply_scheduler(
    scheduler: str,
    base_lr: float,
    current_lr: float,
    step_index: int,
    params: SchedulerParams,
    losses: Sequence[float],
    total_steps: int,
) -> float:
    if scheduler == "None":
        return base_lr
    if scheduler == "Power":
        decay_steps = float(params.get("power_steps", 10.0))
        exponent = float(params.get("power_exponent", 1.0))
        return base_lr / (1.0 + step_index / max(decay_steps, 1.0)) ** exponent
    if scheduler == "Exponential":
        gamma = float(params.get("exp_gamma", 0.9))
        return base_lr * (gamma**step_index)
    if scheduler == "Piecewise":
        boundaries = params.get("piecewise_boundaries", [])
        values = params.get("piecewise_values", [])
        for boundary, value in zip(boundaries, values):
            if step_index < boundary:
                return float(value)
        if values:
            return float(values[-1])
        return base_lr
    if scheduler == "Performance":
        if len(losses) < 2:
            return current_lr
        lambda_factor = float(params.get("lambda_factor", 0.5))
        if losses[-1] > losses[-2]:
            return current_lr * lambda_factor
        return current_lr
    if scheduler == "One Cycle":
        max_lr = float(params.get("one_cycle_max_lr", base_lr * 5))
        total = int(params.get("one_cycle_total", total_steps))
        midpoint = total // 2
        if step_index <= midpoint:
            pct = step_index / max(midpoint, 1)
            return base_lr + (max_lr - base_lr) * pct
        pct = (step_index - midpoint) / max(total - midpoint, 1)
        return max_lr - (max_lr - base_lr) * pct
    return base_lr


def clip_position(position: np.ndarray, objective: ObjectiveFunction) -> np.ndarray:
    x = np.clip(position[0], objective.x_range[0], objective.x_range[1])
    y = np.clip(position[1], objective.y_range[0], objective.y_range[1])
    return np.array([x, y], dtype=float)


def run_optimizer(
    optimizer: str,
    objective: ObjectiveFunction,
    steps: int,
    lr: float,
    beta1: float,
    beta2: float,
    start: np.ndarray,
    scheduler: str,
    scheduler_params: SchedulerParams,
    custom_noise: float,
    custom_seed: int,
) -> OptimizerResult:
    positions = [start.astype(float)]
    losses = [objective.function(*start)]
    momentum = np.zeros(2)
    s = np.zeros(2)
    rms = np.zeros(2)
    m = np.zeros(2)
    v = np.zeros(2)
    current_lr = lr
    rng = np.random.default_rng(custom_seed)

    for step_index in range(1, steps + 1):
        current_lr = apply_scheduler(
            scheduler,
            lr,
            current_lr,
            step_index,
            scheduler_params,
            losses,
            steps,
        )
        position = positions[-1].copy()

        if optimizer == "Nesterov":
            lookahead = position + beta1 * momentum
            gradient = objective.gradient(*lookahead)
        else:
            gradient = objective.gradient(*position)

        if optimizer == "Regular":
            position = position - current_lr * gradient
        elif optimizer == "Momentum":
            momentum = beta1 * momentum - current_lr * gradient
            position = position + momentum
        elif optimizer == "Nesterov":
            momentum = beta1 * momentum - current_lr * gradient
            position = position + momentum
        elif optimizer == "AdaGrad":
            s = s + gradient**2
            position = position - current_lr * gradient / (np.sqrt(s) + EPSILON)
        elif optimizer == "RMSProp":
            rms = beta1 * rms + (1.0 - beta1) * gradient**2
            position = position - current_lr * gradient / (np.sqrt(rms) + EPSILON)
        elif optimizer == "Adam":
            m = beta1 * m + (1.0 - beta1) * gradient
            v = beta2 * v + (1.0 - beta2) * gradient**2
            m_hat = m / (1.0 - beta1**step_index)
            v_hat = v / (1.0 - beta2**step_index)
            position = position - current_lr * m_hat / (np.sqrt(v_hat) + EPSILON)
        elif optimizer == "Custom":
            noise = rng.normal(0.0, custom_noise, size=2)
            position = position - current_lr * gradient + noise
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        position = clip_position(position, objective)
        positions.append(position)
        losses.append(objective.function(*position))

    return OptimizerResult(
        name=optimizer, positions=np.vstack(positions), losses=losses
    )


def render_scheduler_controls(scheduler: str, steps: int) -> SchedulerParams:
    params: SchedulerParams = {}
    if scheduler == "Power":
        params["power_steps"] = st.sidebar.number_input(
            "Decay Steps", value=10.0, min_value=1.0
        )
        params["power_exponent"] = st.sidebar.number_input(
            "Exponent", value=1.0, min_value=0.1
        )
    elif scheduler == "Exponential":
        params["exp_gamma"] = st.sidebar.slider(
            "Gamma", min_value=0.1, max_value=0.999, value=0.9
        )
    elif scheduler == "Piecewise":
        raw_boundaries = st.sidebar.text_input(
            "Boundaries (comma-separated)", value="10,20"
        )
        raw_values = st.sidebar.text_input(
            "Values (comma-separated)", value="0.1,0.05,0.01"
        )
        try:
            params["piecewise_boundaries"] = [
                int(item) for item in raw_boundaries.split(",") if item.strip()
            ]
            params["piecewise_values"] = parse_float_sequence(raw_values)
        except ValueError:
            st.sidebar.warning(
                "Invalid boundaries or values. Falling back to defaults."
            )
            params["piecewise_boundaries"] = [10, 20]
            params["piecewise_values"] = [0.1, 0.05, 0.01]
    elif scheduler == "Performance":
        params["lambda_factor"] = st.sidebar.slider(
            "Lambda Factor", min_value=0.05, max_value=1.0, value=0.5
        )
    elif scheduler == "One Cycle":
        params["one_cycle_max_lr"] = st.sidebar.number_input(
            "Max LR", value=0.5, min_value=0.001
        )
        params["one_cycle_total"] = st.sidebar.slider(
            "Cycle Steps", min_value=1, max_value=steps, value=max(steps // 2, 1)
        )
    return params


def plot_surface(
    objective: ObjectiveFunction, results: Sequence[OptimizerResult]
) -> go.Figure:
    x, y, z = objective.surface()
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale="Viridis",
            opacity=0.7,
            showscale=False,
            name=objective.name,
        )
    )
    for result in results:
        fig.add_trace(
            go.Scatter3d(
                x=result.positions[:, 0],
                y=result.positions[:, 1],
                z=result.losses,
                mode="lines+markers",
                name=result.name,
                line=dict(width=4),
                marker=dict(size=5),
            )
        )
    fig.update_layout(
        title=f"3D Surface and Optimization Paths – {objective.name}",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
        height=600,
    )
    return fig


def plot_contour(
    objective: ObjectiveFunction, results: Sequence[OptimizerResult]
) -> go.Figure:
    x, y, z = objective.surface(resolution=300)
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=x[0], y=y[:, 0], z=z, colorscale="Viridis", contours_coloring="heatmap"
        )
    )
    for result in results:
        fig.add_trace(
            go.Scatter(
                x=result.positions[:, 0],
                y=result.positions[:, 1],
                mode="lines+markers",
                name=result.name,
                line=dict(width=3),
                marker=dict(size=6),
            )
        )
    fig.update_layout(
        title="Top-Down View of Optimization Paths",
        xaxis_title="x",
        yaxis_title="y",
        height=600,
    )
    return fig


def plot_losses(results: Sequence[OptimizerResult]) -> go.Figure:
    fig = go.Figure()
    for result in results:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(result.losses))),
                y=result.losses,
                mode="lines+markers",
                name=result.name,
            )
        )
    fig.update_layout(
        title="Loss over Steps", xaxis_title="Step", yaxis_title="f(x, y)"
    )
    return fig


def summarize_results(results: Sequence[OptimizerResult]) -> pd.DataFrame:
    summary = []
    for result in results:
        final_position = result.positions[-1]
        summary.append(
            {
                "Optimizer": result.name,
                "Final X": round(float(final_position[0]), 4),
                "Final Y": round(float(final_position[1]), 4),
                "Final Loss": round(float(result.losses[-1]), 6),
                "Steps": len(result.losses) - 1,
            }
        )
    return pd.DataFrame(summary)


def main() -> None:
    st.set_page_config(page_title="Gradient Descent Visualizer", layout="wide")
    st.title("Gradient Descent Optimizer Visualizer")
    st.caption("Interactively compare optimizers, learning rates, and schedulers.")

    objectives = get_objectives()

    st.sidebar.header("Simulation Setup")
    objective_name = st.sidebar.selectbox("Objective Function", list(objectives.keys()))
    objective = objectives[objective_name]

    start_x = st.sidebar.slider(
        "Starting x",
        min_value=int(objective.x_range[0]),
        max_value=int(objective.x_range[1]),
        value=5,
    )
    start_y = st.sidebar.slider(
        "Starting y",
        min_value=int(objective.y_range[0]),
        max_value=int(objective.y_range[1]),
        value=5,
    )
    steps = st.sidebar.slider("Steps", min_value=5, max_value=100, value=20)
    base_lr = st.sidebar.number_input(
        "Base Learning Rate", value=0.1, min_value=0.0001, step=0.01, format="%0.4f"
    )

    beta1 = st.sidebar.slider(
        "Beta1 / Momentum", min_value=0.0, max_value=0.999, value=0.9
    )
    beta2 = st.sidebar.slider(
        "Beta2 (Adam)", min_value=0.0, max_value=0.999, value=0.999
    )

    scheduler = st.sidebar.selectbox("Learning Rate Scheduler", SCHEDULER_OPTIONS)
    scheduler_params = render_scheduler_controls(scheduler, steps)

    optimizers = st.sidebar.multiselect(
        "Optimizers", OPTIMIZER_OPTIONS, default=["Regular", "Momentum"]
    )
    custom_noise = st.sidebar.slider(
        "Custom Optimizer Noise", min_value=0.0, max_value=1.0, value=0.05
    )
    custom_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)

    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button("Run Optimization", type="primary")

    st.subheader("Objective Details")
    st.write(objective.description)

    if not optimizers:
        st.info("Select at least one optimizer to start the simulation.")
        return

    if run_simulation:
        start = np.array([start_x, start_y], dtype=float)
        results = [
            run_optimizer(
                optimizer,
                objective,
                steps,
                base_lr,
                beta1,
                beta2,
                start,
                scheduler,
                scheduler_params,
                custom_noise,
                custom_seed,
            )
            for optimizer in optimizers
        ]

        surface_fig = plot_surface(objective, results)
        contour_fig = plot_contour(objective, results)
        losses_fig = plot_losses(results)
        summary_df = summarize_results(results)

        col1, col2 = st.columns(2)
        col1.plotly_chart(surface_fig, use_container_width=True)
        col2.plotly_chart(contour_fig, use_container_width=True)
        st.plotly_chart(losses_fig, use_container_width=True)
        st.dataframe(summary_df, use_container_width=True)


if __name__ == "__main__":
    main()
