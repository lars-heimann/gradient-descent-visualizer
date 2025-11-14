"""Optimization algorithms used in the visualizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .objectives import ObjectiveFunction
from .schedulers import SchedulerParams, apply_scheduler

OPTIMIZER_OPTIONS = [
    "Regular",
    "Momentum",
    "Nesterov",
    "AdaGrad",
    "RMSProp",
    "Adam",
    "Custom",
]

EPSILON = 1e-8


@dataclass
class OptimizerResult:
    """Stores the optimization trajectory of an optimizer run."""

    name: str
    positions: np.ndarray
    losses: List[float]


def clip_position(position: np.ndarray, objective: ObjectiveFunction) -> np.ndarray:
    """Clip a position to the boundaries of the objective function."""

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
    """Execute a single optimization run and collect its trajectory."""

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
