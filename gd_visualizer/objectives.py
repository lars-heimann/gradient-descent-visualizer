"""Objective function definitions for the gradient descent visualizer."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

ScalarFunction = Callable[[float, float], float]
GradientFunction = Callable[[float, float], np.ndarray]


@dataclass(frozen=True)
class ObjectiveFunction:
    """Container for an objective function and its metadata."""

    name: str
    description: str
    function: ScalarFunction
    gradient: GradientFunction
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    def surface(
        self, resolution: int = 200
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a grid for Plotly visualizations."""

        x = np.linspace(self.x_range[0], self.x_range[1], resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], resolution)
        grid_x, grid_y = np.meshgrid(x, y)
        vectorized = np.vectorize(self.function)
        grid_z = vectorized(grid_x, grid_y)
        return grid_x, grid_y, grid_z


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


def get_objectives() -> dict[str, ObjectiveFunction]:
    """Return all available objective functions keyed by their display name."""

    return {
        obj.name: obj
        for obj in (
            steep_valley_function(),
            symmetric_convex_function(),
            asymmetric_convex_function(),
            l_shaped_function(),
        )
    }
