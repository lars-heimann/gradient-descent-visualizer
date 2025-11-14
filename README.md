## Gradient Descent Visualizer

An interactive Streamlit app for exploring gradient-based optimizers, learning rates, and schedulers on various 2D objective functions. The UI lets you compare trajectories, inspect loss curves, and reason about optimizer behavior without touching the underlying code.

### Features
- Multiple objective functions (steep, symmetric, asymmetric, and L-shaped valleys)
- Seven optimizers: Regular GD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, and a noise-driven Custom optimizer
- Learning-rate schedulers (power, exponential, piecewise, performance-based, and one-cycle)
- Interactive 3D surface, 2D contour, and loss plots powered by Plotly
- Download-free deployment with `streamlit` (e.g., Streamlit Community Cloud or any PaaS)

### Quick Start
```bash
uv sync  # or: pip install -e .
uv run streamlit run main.py
```
The app launches in your browser (default: http://localhost:8501). Use the sidebar to select objectives, optimizers, learning-rate settings, and scheduler parameters.

### Development Workflow
Set up the development tooling and git hooks to keep formatting and linting consistent:

```bash
uv sync --group dev
uv run pre-commit install
pre-commit run --all-files
```

- `uv sync --group dev` installs the optional development dependencies defined in `pyproject.toml` (including `pre-commit` and `ruff`).
- `uv run pre-commit install` registers the git hooks for this repository.
- `pre-commit run --all-files` checks and auto-formats the entire codebase; the hooks will run automatically on future commits.

### Controls Overview
- **Objective Function**: Picks the landscape and defines the clipping range for trajectories.
- **Starting x/y & Steps**: Control the initial point and number of gradient steps.
- **Base Learning Rate**: Baseline step size, adjusted further by the chosen scheduler.
- **Beta1/Beta2**: Shared momentum coefficients for the optimizers that use them.
- **Scheduler Settings**: Contextual inputs appear depending on the scheduler type.
- **Optimizers**: Multi-select; run several algorithms simultaneously for side-by-side comparisons.
- **Custom Noise & Seed**: Adds reproducible stochasticity to the Custom optimizer.

### Deploying on the Web
1. Push this repo to a public GitHub repository.
2. Head to [share.streamlit.io](https://share.streamlit.io/), point it to `main.py`, and deploy.
3. Configure secrets/environment variables if you add more advanced features later.

### Development Notes
- The whole app lives in `main.py` with type hints and docstrings for clarity.
- Update `pyproject.toml` if you add new dependencies; `uv sync` keeps the lockfile up-to-date.
- Feel free to extend `OBJECTIVES`, `OPTIMIZER_OPTIONS`, or schedulers—helper utilities centralize most shared logic.
