"""Run simple nandir inversions using the generated SECS synthetic data.

This script is intentionally plain. It loads the arrays saved by
``generate_synth_data.py``, builds the inverse problem

    d = G m

from the magnetometer magnetic-field components, and compares a few solve paths
provided by nandir. The figure is a visual smoke test: the recovered SECS
amplitudes and predicted vertical magnetic field should look plausible compared
with the synthetic truth.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from nandir import LinearInverseProblem, QuadraticRegularization


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synth_data"
FIGURES_DIR = REPO_ROOT / "figures"


def load_synth_data(path: Path) -> dict[str, np.ndarray | sparse.spmatrix]:
    """Load the per-array synthetic dataset written by generate_synth_data.py."""

    dense_names = [
        "m_true",
        "GeB",
        "GnB",
        "GuB",
        "Be",
        "Bn",
        "Bu",
        "xi",
        "eta",
        "xi_model",
        "eta_model",
        "GeB_mag",
        "GnB_mag",
        "GuB_mag",
        "Be_mag",
        "Bn_mag",
        "Bu_mag",
    ]
    arrays: dict[str, np.ndarray | sparse.spmatrix] = {}
    for name in dense_names:
        with np.load(path / f"{name}.npz") as data:
            arrays[name] = data["data"]
    arrays["LTL"] = sparse.load_npz(path / "LTL.npz")
    return arrays


synth_data = load_synth_data(DATA_DIR)

# ``xi``/``eta`` describe the magnetic-field evaluation mesh. ``xi_model`` and
# ``eta_model`` describe the SECS model grid, which is one cell smaller in each
# dimension.
xi, eta = synth_data['xi'], synth_data['eta']
xi_model, eta_model = synth_data['xi_model'], synth_data['eta_model']
m_true = synth_data['m_true']
model_shape = xi_model.shape

GeB, GnB, GuB = synth_data['GeB'], synth_data['GnB'], synth_data['GuB']
Be, Bn, Bu = synth_data['Be'], synth_data['Bn'], synth_data['Bu']

GeB_mag, GnB_mag, GuB_mag = synth_data['GeB_mag'], synth_data['GnB_mag'], synth_data['GuB_mag']

Be_mag, Bn_mag, Bu_mag = synth_data['Be_mag'], synth_data['Bn_mag'], synth_data['Bu_mag']

LTL = synth_data['LTL']


G = np.vstack((GeB_mag, GnB_mag, GuB_mag)) # shape: (3*N, grid.size)
d = np.hstack((Be_mag, Bn_mag, Bu_mag))    # shape: (3*N)

GTG = G.T@G
GTd = G.T@d

scale_gtg = np.median(np.diag(GTG))
scale_ltl = np.median(LTL.diagonal())

l1 = 1e0
l2 = 1e1

reg = (
    l1 * scale_gtg * sparse.eye(GTG.shape[0], format="csr")
    + l2 * scale_gtg / scale_ltl * LTL
)

# Solve the same problem several ways. ``m_precomputed`` should match
# ``m_reg`` exactly because it reuses the same normal equations. ``m_sparse``
# should also match, but it exercises sparse-matrix support.
problem = LinearInverseProblem(G, d)
m_lstsq = problem.solve().model
m_reg = problem.solve(reg=QuadraticRegularization(reg)).model
m_precomputed = problem.solver_from_normal_equations(GTG=GTG, GTd=GTd, reg=QuadraticRegularization(reg)).solve()
m_sparse = LinearInverseProblem(sparse.csr_matrix(G), d).solve(reg=QuadraticRegularization(reg)).model

print(
    "regularized/precomputed relative difference:",
    np.linalg.norm(m_reg - m_precomputed) / np.linalg.norm(m_reg),
)
print(
    "regularized/sparse relative difference:",
    np.linalg.norm(m_reg - m_sparse) / np.linalg.norm(m_reg),
)

Bu_lstsq = GeB.dot(m_lstsq).reshape(xi.shape)
Bu_reg = GeB.dot(m_reg).reshape(xi.shape)
Bu_sparse = GeB.dot(m_sparse).reshape(xi.shape)

FIGURES_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (16, 8), constrained_layout=True)

model_lim = np.abs(m_true).max()
field_lim = np.abs(Bu).max()

models = [
    ('truth', m_true),
    ('least squares', m_lstsq),
    ('regularized', m_reg),
    ('sparse regularized', m_sparse),
]
fields = [
    ('truth', Bu),
    ('least squares', Bu_lstsq),
    ('regularized', Bu_reg),
    ('sparse regularized', Bu_sparse),
]

for i, (title, model) in enumerate(models):
    axes[0, i].pcolormesh(xi_model, eta_model, model.reshape(model_shape), cmap = plt.cm.bwr, vmin = -model_lim, vmax = model_lim, shading='auto')
    axes[0, i].set_title(title)
    axes[0, i].set_aspect('equal')

for i, (title, field) in enumerate(fields):
    axes[1, i].pcolormesh(xi, eta, field, cmap = plt.cm.bwr, vmin = -field_lim, vmax = field_lim, shading='auto')
    axes[1, i].set_title(title)
    axes[1, i].set_aspect('equal')

axes[0, 0].set_ylabel('SECS amplitudes')
axes[1, 0].set_ylabel('Ground magnetic field')

plt.savefig(FIGURES_DIR / 'synth_truth.png', dpi=200)
