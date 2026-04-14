# nandir

`nandir` is a small Python package for linear inverse problems. It is built for
the kind of workflow where you repeatedly write the same normal-equation code:

```text
d = G m
(G.T W G + R) m = G.T W d
```

Here `G` maps model parameters to data, `d` is the data vector, `W` is an
optional data weighting matrix, and `R` is a quadratic regularization matrix.

The package exposes three main objects:

- `Solver`: low-level normal-equation solver
- `LinearInverseProblem`: higher-level wrapper for repeated solves
- `QuadraticRegularization`: small container for regularization terms

## Install

For library use:

```bash
pip install -e .
```

For the SECS example scripts:

```bash
pip install -e ".[examples]"
```

## Basic Usage

```python
import numpy as np

from nandir import LinearInverseProblem, QuadraticRegularization

G = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
d = np.array([1.0, 2.0, 1.0])
w = np.array([1.0, 2.0, 1.0])

reg = QuadraticRegularization(np.eye(2), lreg=0.1)

problem = LinearInverseProblem(G, d)
result = problem.solve(w=w, reg=reg)
m = result.model
```

Use `LinearInverseProblem` when you want a simple interface and diagnostics.
Use `Solver` directly when you already have `GTG` and `GTd`, or when you need
posterior covariance and resolution quantities.

## Weights

`w` controls data weighting:

- `w=None`: unweighted least squares
- 1D array: interpreted as the diagonal of `W`
- 2D dense array: interpreted as the full `W`
- 2D sparse matrix: interpreted as the full sparse `W`

This means these two are equivalent:

```python
from nandir import Solver

weights = np.array([1.0, 2.0, 1.0])
Solver(G=G, d=d, w=weights)
Solver(G=G, d=d, w=np.diag(weights))
```

## Regularization

`QuadraticRegularization(LTL, lreg=...)` adds `lreg * LTL` to the normal
equations. If `scale=True`, `nandir` rescales the term by typical diagonal
values so that a regularization value is less tied to the raw units of the
matrices:

```python
reg = [
    QuadraticRegularization(np.eye(G.shape[1]), lreg=1e-3, scale=True),
    QuadraticRegularization(LTL, lreg=1e-2, scale=True),
]

m = LinearInverseProblem(G, d).solve(reg=reg).model
```

## Repeated Solves

When only the regularization changes, reuse `GTG` and `GTd`:

```python
problem = LinearInverseProblem(G, d)
GTG, GTd = problem.normal_equations()

solver = problem.solver_from_normal_equations(
    GTG=GTG,
    GTd=GTd,
    reg=QuadraticRegularization(LTL, lreg=1e-2),
)
m = solver.solve()
```

For regularization sweeps:

```python
results = problem.regularization_path(
    lambdas=np.logspace(-4, -1, 8),
    LTL=LTL,
    scale=True,
)
```

For robust fitting:

```python
irls = problem.huber_irls(delta=1.5, max_iterations=10)
robust_model = irls.solution
```

## Sparse Matrices

`G`, `GTG`, `LTL`, and full 2D weights may be SciPy sparse matrices. Sparse
systems use sparse factorization for solves. For posterior work on sparse
systems, prefer targeted quantities:

- `posterior_covariance_columns(indices)` for selected posterior columns
- `posterior_covariance_diagonal()` for exact posterior variances
- `posterior_covariance_sparse(drop_tol=...)` for a thresholded sparse approximation

The exact inverse of a sparse matrix is usually dense, so use full
`posterior_covariance` only for small problems.

## SECS Example

The example scripts mirror the structure of the original `secsy` SECS example.
The generator builds a synthetic SECS amplitude pattern from the SciPy raccoon
image, evaluates the magnetic field, samples that field at SuperMAG station
locations, and stores only the arrays needed for the inversion examples.

Generate data:

```bash
python scripts/generate_synth_data.py
```

The generated arrays are stored independently in `data/synth_data/`. Dense
arrays are compressed `.npz` files, while `LTL` is stored as a sparse SciPy
matrix. The old single-file cache `data/synth_data.npz` is ignored.

Run the example inversions and make the validation figure:

```bash
python scripts/examples.py
```

The example compares:

- an unregularized least-squares solution
- a dense regularized solve
- the same solve using precomputed normal equations
- the same solve using a sparse forward matrix

The printed relative differences should be near numerical precision for the
equivalent dense/precomputed/sparse solves.
