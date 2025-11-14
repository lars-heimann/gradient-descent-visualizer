"""Plotting utilities for the gradient descent visualizer."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from .objectives import ObjectiveFunction
from .optimizers import OptimizerResult


def plot_surface(
    objective: ObjectiveFunction, results: Sequence[OptimizerResult]
) -> go.Figure:
    """Return a 3D surface plot with optimization trajectories."""

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
    """Return a top-down contour plot with trajectories."""

    x, y, z = objective.surface(resolution=300)
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=x[0],
            y=y[:, 0],
            z=z,
            colorscale="Viridis",
            contours_coloring="heatmap",
            showscale=False,
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def plot_losses(results: Sequence[OptimizerResult]) -> go.Figure:
    """Return a line plot summarizing the loss over time for each optimizer."""

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
    """Return a human-readable summary of the final optimizer states."""

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
