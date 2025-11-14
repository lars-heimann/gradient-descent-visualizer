"""Learning rate scheduler utilities and Streamlit controls."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

SchedulerParams = dict[str, object]

SCHEDULER_OPTIONS = [
    "None",
    "Power",
    "Exponential",
    "Piecewise",
    "Performance",
    "One Cycle",
]


def parse_float_sequence(raw_value: str) -> list[float]:
    """Parse a comma-separated string into a list of floats."""

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
    """Apply the selected scheduler and return the learning rate for this step."""

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


def render_scheduler_controls(scheduler: str, steps: int) -> SchedulerParams:
    """Render Streamlit sidebar controls for the selected scheduler."""

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
