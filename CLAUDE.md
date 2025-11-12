# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepXDE is a library for scientific machine learning and physics-informed learning, implementing:
- Physics-Informed Neural Networks (PINN) for solving forward/inverse ODEs/PDEs/IDEs/fPDEs
- Deep Operator Networks (DeepONet) for learning operators
- Multifidelity Neural Networks (MFNN) for learning from multifidelity data

The library supports 5 tensor backends: TensorFlow 1.x (tensorflow.compat.v1), TensorFlow 2.x, PyTorch, JAX, and PaddlePaddle. PaddlePaddle is currently the recommended backend.

## Backend Selection

DeepXDE uses a **backend abstraction layer** that allows code to run on different deep learning frameworks. The backend is selected at runtime:

1. Via environment variable: `set DDE_BACKEND=pytorch` (or `export DDE_BACKEND=pytorch` on Unix)
2. Via config file: `~/.deepxde/config.json`
3. Auto-detection if no backend is specified

Backend selection happens in `deepxde/backend/__init__.py:load_backend()` which dynamically imports the appropriate backend module. All backend-specific implementations are in `deepxde/backend/{backend_name}/` and `deepxde/nn/{backend_name}/`, `deepxde/optimizers/{backend_name}/`, etc.

**Important**: Backend must be imported before any other DeepXDE modules (see `deepxde/__init__.py:21`).

## Core Architecture

### Model Training Workflow

1. **Define Problem**: Create geometry + PDE/BC/IC
   - `deepxde.geometry.*` - Domain geometries (Interval, Rectangle, Disk, CSG operations, etc.)
   - `deepxde.icbc.*` - Initial/boundary conditions (DirichletBC, NeumannBC, RobinBC, PeriodicBC, IC)

2. **Create Data**: Wrap problem into a Data object
   - `deepxde.data.PDE` - For ODE/PDE problems
   - `deepxde.data.TimePDE` - For time-dependent PDEs
   - `deepxde.data.FPDE`/`TimeFPDE` - For fractional PDEs
   - `deepxde.data.IDE` - For integro-differential equations
   - `deepxde.data.Function` - For function approximation
   - `deepxde.data.PDEOperator` - For operator learning (DeepONet)

3. **Build Network**: Select neural network architecture
   - `deepxde.nn.FNN` - Fully connected neural network (main implementation)
   - `deepxde.nn.DeepONet` - Deep operator network
   - `deepxde.nn.MIONet` - Multiple-input operator network
   - `deepxde.nn.MFNN` - Multifidelity neural network
   - Network implementations are backend-specific in `deepxde/nn/{backend}/`

4. **Create Model**: Combine Data + Network
   - `deepxde.Model(data, net)` - Main model class in `deepxde/model.py`

5. **Compile & Train**: Configure optimizer and train
   - `model.compile()` - Set optimizer, learning rate, loss, metrics
   - `model.train()` - Train with specified iterations

### Key Components

- **Geometry** (`deepxde/geometry/`): Abstract base class `Geometry` with methods `inside()`, `on_boundary()`, `random_points()`, `uniform_points()`. Supports CSG operations (union, difference, intersection) via `deepxde/geometry/csg.py`.

- **Gradients** (`deepxde/gradients/`): Three AD methods available:
  - Reverse mode (default, backpropagation)
  - Forward mode
  - Zero Coordinate Shift (ZCS) in `deepxde/zcs/`
  - Set via `dde.config.set_default_autodiff("reverse"/"forward")`

- **Config** (`deepxde/config.py`): Global configuration
  - `set_default_float()` - float16/float32/float64/mixed precision
  - `set_random_seed()` - Reproducibility
  - `enable_xla_jit()`/`disable_xla_jit()` - XLA compilation
  - `set_parallel_scaling()` - Weak/strong scaling for data parallel

- **Callbacks** (`deepxde/callbacks.py`): Training callbacks like early stopping, model checkpointing, etc.

## Running Examples

Examples are organized by problem type:
- `examples/pinn_forward/` - Forward PDE problems
- `examples/pinn_inverse/` - Inverse/parameter identification problems
- `examples/operator/` - Operator learning (DeepONet)
- `examples/function/` - Function approximation

To run an example:
```bash
cd examples/pinn_forward
python Poisson_Dirichlet_1d.py
```

Each example is self-contained and demonstrates the full workflow. Examples include backend-specific code sections (commented out) for different frameworks.

## Testing

Backend validation is performed through the example suites themselves. After installing
the desired backend, run a representative example (e.g.,
`python examples/pinn_inverse/force_estimation.py --help`) or your project-specific
integration tests. This keeps the repository focused on production code while letting
teams plug into their preferred unit/integration frameworks.

## Development Notes

### Adding New Neural Networks

1. Implement network in `deepxde/nn/{backend}/` for each supported backend
2. Add to `deepxde/nn/{backend}/__init__.py` exports
3. The module loader will make it available as `dde.nn.YourNetwork`

### Adding New Geometries

1. Inherit from `deepxde.geometry.Geometry` base class
2. Implement required abstract methods: `inside()`, `on_boundary()`
3. Implement `random_points()`, `uniform_points()` for sampling
4. Add to `deepxde/geometry/__init__.py`

### Backend-Specific Code

When writing backend-agnostic code:
- Use `from deepxde.backend import backend_name, tf, torch, jax, paddle`
- Check `backend_name` for conditional logic
- Use `dde.grad.*` for automatic differentiation (not direct backend calls)
- Tensor operations should use backend-specific functions via the abstraction layer

### Data Parallel Training

DeepXDE supports data parallel training via Horovod (currently only for tensorflow.compat.v1):
- Set `OMPI_COMM_WORLD_SIZE` environment variable to enable
- Use `dde.config.set_parallel_scaling("weak"/"strong")`
- See configuration in `deepxde/config.py:11-38`

## Common Patterns

### Defining PDEs
PDEs are defined as Python functions that take `(x, y)` where `x` is input coordinates and `y` is the network output:

```python
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)  # Second derivative
    return -dy_xx - source_term(x)
```

### Boundary Conditions
BCs are defined with a geometry, a function, and a filter:

```python
bc = dde.icbc.DirichletBC(geometry, lambda x: np.sin(x), lambda x, on_boundary: on_boundary)
```

### Mixed Precision Training
For faster training with lower memory:

```python
dde.config.set_default_float("mixed")
model.compile("adam", lr=0.001)
```

### Model Checkpointing
Save models during training:

```python
checkpointer = dde.callbacks.ModelCheckpoint("model/model", save_better_only=True)
model.train(iterations=10000, callbacks=[checkpointer])
```

## Installation

Install a backend first (e.g., `pip install torch` or `pip install paddlepaddle`), then:
- Via pip: `pip install deepxde`
- Via conda: `conda install -c conda-forge deepxde`
- For development: Clone repo and use directly (add to PYTHONPATH or work in parent directory)

## Documentation

Full documentation at: https://deepxde.readthedocs.io
- API reference: https://deepxde.readthedocs.io/en/latest/modules/deepxde.html
- Demos: https://deepxde.readthedocs.io/en/latest/demos/

## Project Module: estimation (Force Estimation PINN)

The repo contains a project-specific package `estimation/` implementing a PINN
to estimate interaction forces in a quadrotor–payload system.

Key modules
- `estimation/cli.py`: CLI entrypoint (preferred entry). Wrapper at `estimation/force_estimation.py`.
- `estimation/physics.py`: ResidualContext/ResidualScaler and residual assembly.
- `estimation/data_utils.py`: Dataset I/O, Normalizer, ConstantManager, quantile clipping, caches.
- `estimation/geometry_utils.py`: Mixed sampling, visualization, auto num_domain.
- `estimation/training.py`: Network/BC builders, training routines, evaluate/export.
- `estimation/metrics_logger.py`: Stepwise CSV logging of losses, per-component MSE/RMS, weights, grad L2.
- `estimation/gradient_monitor.py`: Global gradient L2 monitor (PyTorch), CSV logging.
- `estimation/weight_scheduler.py`: Adaptive weight scheduler (EMA of residual RMS).
- `examples/pinn_inverse/*`: Re-export shims importing `estimation` so legacy paths still work.

Data
- Default dataset path: `estimation/datasets/force_estimation_train.csv`
- Override via `--data` or `FORCE_ESTIMATION_DATA` env var.

Run a quick demo (PyTorch)
```
DDE_BACKEND=pytorch python3 estimation/force_estimation.py \
  --demo --seed 42 --sample-size 64 --max-points 64 \
  --adam-iters 5 --disable-lbfgs --num-domain 64 --skip-unit-check \
  --metrics-csv logs/training_metrics.csv \
  --residual-variance-path logs/residual_variance.csv \
  --grad-monitor
```

Stage 3 features exposed via CLI
- Residual normalization: `--residual-norm-mode {off,median,mean}`, `--residual-scale-r{1,2,3}`
- Static weights: `--loss-weight-mode {manual,inverse_residual}`, `--loss-weight-r{1,2,3}`, `--bc-loss-weight`
- Adaptive weights: `--adaptive-weights`, `--adaptive-weight-{period,alpha,min,max}`
- Monitoring: `--metrics-csv`, `--grad-monitor{,-period,-log,-patience,-min,-max}`

Testing
- `python3 -m pytest tests`
- The example shims can be invoked (e.g., `python examples/pinn_inverse/force_estimation.py --help`),
  but prefer the `estimation/` entry for development.
