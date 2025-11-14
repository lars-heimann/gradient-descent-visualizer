"""Streamlit application to visualize gradient descent optimizers."""

from __future__ import annotations

import numpy as np
import streamlit as st

from gd_visualizer.objectives import get_objectives
from gd_visualizer.optimizers import OPTIMIZER_OPTIONS, run_optimizer
from gd_visualizer.plotting import (
    plot_contour,
    plot_losses,
    plot_surface,
    summarize_results,
)
from gd_visualizer.schedulers import SCHEDULER_OPTIONS, render_scheduler_controls


def main() -> None:
    """Render the Streamlit application."""

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

    beta1 = st.sidebar.slider("Beta1 / Momentum", min_value=0.0, max_value=0.999, value=0.9)
    beta2 = st.sidebar.slider("Beta2 (Adam)", min_value=0.0, max_value=0.999, value=0.999)

    scheduler = st.sidebar.selectbox("Learning Rate Scheduler", SCHEDULER_OPTIONS)
    scheduler_params = render_scheduler_controls(scheduler, steps)

    optimizers = st.sidebar.multiselect(
        "Optimizers", OPTIMIZER_OPTIONS, default=["Regular", "Momentum"]
    )
    custom_noise = st.sidebar.slider(
        "Custom Optimizer Noise", min_value=0.0, max_value=1.0, value=0.05
    )
    custom_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)

    st.subheader("Objective Details")
    st.write(objective.description)

    if not optimizers:
        st.info("Select at least one optimizer to start the simulation.")
        return

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
