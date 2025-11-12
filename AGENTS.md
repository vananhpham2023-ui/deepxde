# AGENTS.md

This guide helps AI coding agents work effectively in this repository, with a
focus on the project-specific estimation package for the force estimation PINN.

Scope: unless otherwise noted, these instructions apply to the entire repo.

Goals
- Keep upstream DeepXDE code intact; add project code under `estimation/`.
- Preserve backward compatibility for `examples/pinn_inverse/*` via re-export shims.
- Prefer small, surgical changes over sweeping refactors.

Directory Layout (project-specific)
- `estimation/cli.py`: Main CLI entry that orchestrates data → geometry → physics → training.
- `estimation/force_estimation.py`: Thin wrapper exposing `main()` from `cli.py`.
- `estimation/physics.py`: ResidualContext, ResidualScaler, residual assembly, NumPy diagnostics.
- `estimation/data_utils.py`: Normalizer, ConstantManager, dataset loading, quantile clipping, caches.
- `estimation/geometry_utils.py`: Mixed sampling, visualization, geometry construction, auto num_domain.
- `estimation/training.py`: Network and BC builders, training/evaluation/export helpers.
- `estimation/metrics_logger.py`: CSV logger for losses, per-component MSE/RMS, loss weights, grad L2.
- `estimation/gradient_monitor.py`: Global gradient L2 monitor and CSV logger (PyTorch only).
- `estimation/weight_scheduler.py`: Adaptive weight scheduler (EMA of per-component RMS).
- `examples/pinn_inverse/*.py`: Re-export shims that import from `estimation/*`.

Key Conventions
- New functionality should live under `estimation/` and be wired into `estimation/cli.py`.
- Do not break legacy imports: keep re-export shims under `examples/pinn_inverse/` in sync.
- Use argparse flags in `estimation/cli.py` for new toggles; keep defaults conservative.
- Write Chinese-facing user messages where appropriate; keep code comments concise.
- Avoid adding global state; pass configuration explicitly between helpers.

Data and Paths
- Default dataset: `estimation/datasets/force_estimation_train.csv`.
- Override via `--data` (file or folder) or `FORCE_ESTIMATION_DATA` environment variable.
- Normalization stats cache: `normalizer_stats.pkl` (can be overridden via CLI).

Monitoring and Training
- Use `MetricsLogger` to log step, total losses, per-component MSE/RMS, loss weights, and optional gradient L2.
- Enable gradient monitoring via `--grad-monitor` (PyTorch only) to log global grad norm to CSV.
- Residual normalization modes: `--residual-norm-mode {off,median,mean}`; manual overrides per group via `--residual-scale-r{1,2,3}`.
- Static loss weights: `--loss-weight-mode {manual,inverse_residual}` plus `--loss-weight-r{1,2,3}` and `--bc-loss-weight`.
- Adaptive weights: `--adaptive-weights` with `--adaptive-weight-{period,alpha,min,max}`.

Testing and Validation
- Run unit/integration tests: `python3 -m pytest tests`.
- Quick demo run (PyTorch backend):
  `DDE_BACKEND=pytorch python3 estimation/force_estimation.py --demo --seed 42 --sample-size 64 --max-points 64 --adam-iters 5 --disable-lbfgs --num-domain 64 --skip-unit-check --metrics-csv logs/training_metrics.csv --residual-variance-path logs/residual_variance.csv --grad-monitor`

PR/Change Guidance
- Keep changes minimal and focused. Do not modify unrelated examples or core DeepXDE code.
- Maintain compatibility of CLI flags; document new flags in README and TASK5.
- If adding modules, include them in `estimation/__init__.py` exports when appropriate.

