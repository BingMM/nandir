"""Generate the synthetic SECS dataset used by ``scripts/examples.py``.

The script follows the first half of the SECS example notebook: create a
synthetic divergence-free SECS amplitude pattern from the SciPy raccoon image,
evaluate the magnetic field on a full grid, sample that field at SuperMAG
station locations, and save only the arrays needed by the nandir examples.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from secsy import cubedsphere as cs
from secsy import get_SECS_B_G_matrices


RE = 6371.2e3
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SYNTH_DATA_DIR = DATA_DIR / "synth_data"
STATION_FILE = DATA_DIR / "20230627-09-32-supermag-stations.csv"


def face_image() -> np.ndarray:
    """Load the raccoon image used by the original secsy example."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from scipy.misc import face

        return face(gray=True)
    except ImportError:
        from scipy.datasets import face

        return face(gray=True)


def load_station_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read geographic station longitude and latitude from the SuperMAG CSV."""

    stations = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        usecols=(1, 2),
        dtype=float,
        encoding="utf-8",
    )
    return stations["GEOLON"], stations["GEOLAT"]


def save_dense(name: str, array) -> None:
    """Save one dense array in the synthetic-data folder."""

    np.savez_compressed(SYNTH_DATA_DIR / f"{name}.npz", data=np.asarray(array, dtype=np.float32))


def save_sparse(name: str, matrix) -> None:
    """Save one sparse matrix in the synthetic-data folder."""

    sparse.save_npz(SYNTH_DATA_DIR / f"{name}.npz", sparse.csr_matrix(matrix, dtype=np.float32))


def save_synth_data(arrays: dict[str, object]) -> None:
    """Save each synthetic-data array as an independent file."""

    SYNTH_DATA_DIR.mkdir(exist_ok=True)
    for name, array in arrays.items():
        if name == "LTL":
            save_sparse(name, array)
        else:
            save_dense(name, array)


# load racoon picture to represent the SECS divergence-free amplitude (the curl of the current)
I = gaussian_filter(face_image(), 3)[100:500:6, 500:-200:6][::-1] # smooth, crop, downsample, and turn upside-down
shp = I.shape
I = I - I.mean()

# set up cubed sphere projection and grid
projection = cs.CSprojection((15, 70), 0) # central (lon, lat) and orientation of the grid
grid = cs.CSgrid(projection, shp[1] * .5e5, shp[0]*.5e5, shp[0], shp[1], R = RE, wshift = 1e3)

# magnetic field on ground (evaluated on the full "mesh" grid):
GeB, GnB, GuB = get_SECS_B_G_matrices(grid.lat_mesh, grid.lon_mesh, RE, grid.lat, grid.lon)

# Use matrices to calculate the actual values:
I = I.flatten()
Be, Bn, Bu = GeB.dot(I).reshape(grid.lat_mesh.shape), GnB.dot(I).reshape(grid.lat_mesh.shape), GuB.dot(I).reshape(grid.lat_mesh.shape)


# find SuperMAG magnetometers that are in the grid:
station_lon, station_lat = load_station_coordinates(STATION_FILE)
iii = grid.ingrid(station_lon, station_lat)
lon_mag, lat_mag = station_lon[iii], station_lat[iii]

# evaluate the magnetic field at these points:
GeB_mag, GnB_mag, GuB_mag = get_SECS_B_G_matrices(lat_mag, lon_mag, RE, grid.lat, grid.lon)
Be_mag, Bn_mag, Bu_mag = GeB_mag.dot(I), GnB_mag.dot(I), GuB_mag.dot(I)

# Regularization matrix for east-west smoothing in the inverse problem.
De, Dn = grid.get_Le_Ln() # get matrices that calculate eastward and northward components of gradient
DTD = De.T.dot(De)

DATA_DIR.mkdir(exist_ok=True)
save_synth_data({
    "m_true": I,
    "GeB": GeB,
    "GnB": GnB,
    "GuB": GuB,
    "Be": Be,
    "Bn": Bn,
    "Bu": Bu,
    "xi": grid.xi_mesh,
    "eta": grid.eta_mesh,
    "xi_model": grid.xi,
    "eta_model": grid.eta,
    "GeB_mag": GeB_mag,
    "GnB_mag": GnB_mag,
    "GuB_mag": GuB_mag,
    "Be_mag": Be_mag,
    "Bn_mag": Bn_mag,
    "Bu_mag": Bu_mag,
    "LTL": DTD,
})

print(f"Wrote {SYNTH_DATA_DIR}")
