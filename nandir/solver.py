"""Low-level tools for linear inverse problems.

The :class:`Solver` class solves inverse problems of the form

    (G.T W G + L.T L) m = G.T W d

where ``G`` maps model parameters to data, ``d`` is the data vector, ``W`` is
an optional data weighting matrix, and ``L.T L`` is one or more quadratic
regularization terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgError, cho_factor, cho_solve, lstsq
from scipy.sparse.linalg import factorized

try:
    import cupy as cp
except ImportError:
    cp = None


DenseArray = np.ndarray
MatrixLike = DenseArray | sparse.spmatrix


@dataclass(frozen=True)
class QuadraticRegularization:
    """Quadratic regularization term added to the normal equations.

    Parameters
    ----------
    LTL
        Matrix for a quadratic penalty ``m.T @ LTL @ m``. This can be dense or
        sparse.
    lreg
        Scalar multiplier for the penalty.
    scale
        If true, rescale ``LTL`` by a typical diagonal value and by the typical
        diagonal value of ``GTG``. This makes ``lreg`` less dependent on the
        absolute units of ``G`` and ``LTL``.
    LTL_scale
        Optional scale value to use when ``scale=True``. If omitted, the median
        diagonal value of ``LTL`` is used.
    """

    LTL: MatrixLike
    lreg: float = 1.0
    scale: bool = False
    LTL_scale: float | None = None


def _is_sparse(value: object) -> bool:
    return sparse.issparse(value)


def _as_data_array(name: str, value: object) -> DenseArray | None:
    if value is None:
        return None

    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array
    raise ValueError(f"{name} must be 1D or 2D, got {array.ndim}D")


def _as_matrix(name: str, value: object) -> MatrixLike | None:
    if value is None:
        return None

    if _is_sparse(value):
        matrix = value.astype(float)
    else:
        matrix = np.asarray(value, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    return matrix


def _as_weight(name: str, value: object) -> DenseArray | MatrixLike | None:
    if value is None:
        return None

    if _is_sparse(value):
        matrix = value.astype(float)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D")
        return matrix

    array = np.asarray(value, dtype=float)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1D or 2D")
    return array


def _normalize_regularization_item(item: object) -> QuadraticRegularization:
    if isinstance(item, QuadraticRegularization):
        return item

    if _is_sparse(item) or isinstance(item, np.ndarray):
        return QuadraticRegularization(LTL=item)

    if hasattr(item, "LTL"):
        return QuadraticRegularization(
            LTL=_as_matrix("LTL", item.LTL),
            lreg=float(getattr(item, "lreg", 1.0)),
            scale=bool(getattr(item, "scale", False)),
            LTL_scale=getattr(item, "LTL_scale", None),
        )

    return QuadraticRegularization(LTL=_as_matrix("LTL", item))


def _normalize_regularization(
    reg: object | Iterable[object] | None,
) -> tuple[QuadraticRegularization, ...]:
    if reg is None:
        return ()

    if isinstance(reg, (list, tuple)):
        items = reg
    else:
        items = (reg,)

    return tuple(_normalize_regularization_item(item) for item in items)


def _diag_values(matrix: MatrixLike) -> DenseArray:
    if _is_sparse(matrix):
        return np.asarray(matrix.diagonal()).reshape(-1)
    return np.diag(np.asarray(matrix))


def _median_diag(matrix: MatrixLike) -> float:
    diagonal = _diag_values(matrix)
    return float(np.median(diagonal))


def _zero_matrix_like(reference: MatrixLike) -> MatrixLike:
    if _is_sparse(reference):
        return sparse.csr_matrix(reference.shape, dtype=float)
    return np.zeros_like(np.asarray(reference), dtype=float)


def _coerce_matrix_like(reference: MatrixLike, value: MatrixLike) -> MatrixLike:
    if _is_sparse(reference):
        return value if _is_sparse(value) else sparse.csr_matrix(np.asarray(value, dtype=float))
    return value.toarray() if _is_sparse(value) else np.asarray(value, dtype=float)


def _weighted_gram_matrix(G: MatrixLike, w: DenseArray | MatrixLike | None) -> MatrixLike:
    if w is None:
        return G.T @ G
    if _is_sparse(w):
        return G.T @ (w @ G)
    if np.asarray(w).ndim == 1:
        weights = np.asarray(w, dtype=float)
        if _is_sparse(G):
            return G.T @ sparse.diags(weights) @ G
        return G.T @ (weights[:, None] * G)
    return G.T @ (np.asarray(w, dtype=float) @ G)


def _weighted_rhs(
    G: MatrixLike,
    d: DenseArray,
    w: DenseArray | MatrixLike | None,
) -> DenseArray:
    if w is None:
        return np.asarray(G.T @ d, dtype=float)
    if _is_sparse(w):
        return np.asarray(G.T @ (w @ d), dtype=float)
    if np.asarray(w).ndim == 1:
        return np.asarray(G.T @ (np.asarray(w, dtype=float) * d), dtype=float)
    return np.asarray(G.T @ (np.asarray(w, dtype=float) @ d), dtype=float)


def _weighted_transpose(G: MatrixLike, w: DenseArray | MatrixLike | None) -> MatrixLike:
    if w is None:
        return G.T
    if _is_sparse(w):
        return G.T @ w
    if np.asarray(w).ndim == 1:
        weights = np.asarray(w, dtype=float)
        if _is_sparse(G):
            return G.T @ sparse.diags(weights)
        return np.asarray(G.T) * weights
    return G.T @ np.asarray(w, dtype=float)


def _solve_with_sparse_factorization(solver_fn, rhs: DenseArray) -> DenseArray:
    if rhs.ndim == 1:
        return np.asarray(solver_fn(rhs), dtype=float)
    columns = [np.asarray(solver_fn(rhs[:, idx]), dtype=float) for idx in range(rhs.shape[1])]
    return np.column_stack(columns)


class Solver:
    """Lazy linear inverse-problem solver based on normal equations.

    This is the low-level class. It is useful when you already know the exact
    matrices you want to solve with, or when you want direct access to derived
    quantities such as ``GTG``, ``GTd``, posterior covariance columns, and the
    model resolution matrix.

    Parameters
    ----------
    G, d
        Forward matrix and data. If supplied, ``GTG`` and ``GTd`` are built
        lazily. ``G`` may be a dense NumPy array or a SciPy sparse matrix.
    w
        Optional data weights. A 1D array is interpreted as the diagonal of a
        weighting matrix. A 2D dense or sparse matrix is used as a full
        weighting matrix.
    GTG, GTd
        Precomputed normal-equation terms. Use these for repeated solves where
        only the regularization changes.
    reg
        One regularization term, a list of terms, or raw ``LTL`` matrices. Raw
        matrices are interpreted as ``QuadraticRegularization(LTL=matrix)``.
    use_gpu
        If CuPy is installed and the system matrix is dense, use CuPy for
        posterior covariance solves.
    **lstsq_kwargs
        Extra keyword arguments passed to ``scipy.linalg.lstsq`` when Cholesky
        factorization is not available.
    """

    def __init__(
        self,
        G: object = None,
        d: object = None,
        w: object = None,
        GTG: object = None,
        GTd: object = None,
        reg: object | Iterable[object] | None = None,
        use_gpu: bool = False,
        **lstsq_kwargs,
    ) -> None:
        self.G = _as_matrix("G", G)
        self.d = _as_data_array("d", d)
        self.w = _as_weight("w", w)
        self.use_gpu = use_gpu
        self.lstsq_kwargs = lstsq_kwargs

        self._GT = None
        self._GTd = _as_data_array("GTd", GTd)
        self._GTG = _as_matrix("GTG", GTG)
        self._GTG_scale = None
        self.reg = _normalize_regularization(reg)
        self._LTL = None
        self._system_matrix = None

        self._c_factor = None
        self._c_valid = None
        self._sparse_solver = None
        self._Cmpost = None
        self._Cmpost_diag = None
        self._Rm = None
        self.m = None

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self._GTG is None and self.G is None:
            raise ValueError("Either G or GTG must be provided")

        if self._GTd is None and (self.G is None or self.d is None):
            raise ValueError("Either GTd or both G and d must be provided")

        if self.G is not None and self.d is not None and self.G.shape[0] != self.d.shape[0]:
            raise ValueError("G and d must have the same number of rows")

        if self.G is not None and self.w is not None:
            rows = self.G.shape[0]
            if _is_sparse(self.w):
                if self.w.shape != (rows, rows):
                    raise ValueError("Sparse w must be a square matrix matching the rows of G")
            elif np.asarray(self.w).ndim == 1 and np.asarray(self.w).shape[0] != rows:
                raise ValueError("1D w must have one entry per row of G")
            elif np.asarray(self.w).ndim == 2 and np.asarray(self.w).shape != (rows, rows):
                raise ValueError("2D w must be a square matrix matching the rows of G")

        if self._GTG is not None and self._GTG.shape[0] != self._GTG.shape[1]:
            raise ValueError("GTG must be square")

        if self._GTG is not None and self._GTd is not None and self._GTG.shape[1] != self._GTd.shape[0]:
            raise ValueError("GTG and GTd have incompatible shapes")

    @property
    def GT(self) -> MatrixLike:
        if self._GT is None:
            if self.G is None:
                raise ValueError("GT cannot be formed without G")
            self._GT = _weighted_transpose(self.G, self.w)
        return self._GT

    @property
    def GTd(self) -> DenseArray:
        if self._GTd is None:
            if self.G is None or self.d is None:
                raise ValueError("GTd cannot be formed without G and d")
            self._GTd = _weighted_rhs(self.G, self.d, self.w)
        return self._GTd

    @property
    def GTG(self) -> MatrixLike:
        if self._GTG is None:
            if self.G is None:
                raise ValueError("GTG cannot be formed without G")
            self._GTG = _weighted_gram_matrix(self.G, self.w)
        return self._GTG

    @property
    def GTG_scale(self) -> float:
        if self._GTG_scale is None:
            self._GTG_scale = _median_diag(self.GTG)
        return self._GTG_scale

    @property
    def LTL(self) -> MatrixLike:
        if self._LTL is None:
            total = _zero_matrix_like(self.GTG)
            for reg in self.reg:
                term = _coerce_matrix_like(self.GTG, _as_matrix("LTL", reg.LTL))
                if term.shape != self.GTG.shape:
                    raise ValueError("Regularization matrix must match the shape of GTG")

                weight = reg.lreg
                if reg.scale:
                    scale = reg.LTL_scale
                    if scale is None:
                        scale = _median_diag(term)
                    if scale == 0:
                        raise ValueError("Regularization scale cannot be zero")
                    total = total + weight * term / scale * self.GTG_scale
                else:
                    total = total + weight * term
            self._LTL = total
        return self._LTL

    @property
    def system_matrix(self) -> MatrixLike:
        if self._system_matrix is None:
            self._system_matrix = self.GTG + self.LTL
        return self._system_matrix

    def _can_use_gpu(self) -> bool:
        return cp is not None and self.use_gpu and not _is_sparse(self.system_matrix)

    def _solve_linear_system(self, rhs: DenseArray, **lstsq_kwargs) -> DenseArray:
        rhs = np.asarray(rhs, dtype=float)

        if _is_sparse(self.system_matrix):
            if self._sparse_solver is None:
                self._sparse_solver = factorized(sparse.csc_matrix(self.system_matrix))
            return _solve_with_sparse_factorization(self._sparse_solver, rhs)

        if self.c_valid:
            return cho_solve(self.c_factor, rhs, check_finite=False)

        return lstsq(self.system_matrix, rhs, **(self.lstsq_kwargs | lstsq_kwargs))[0]

    def _posterior_covariance_blocks(
        self,
        indices: DenseArray,
        *,
        use_gpu: bool = True,
    ) -> DenseArray:
        size = self.system_matrix.shape[0]
        rhs = np.zeros((size, len(indices)), dtype=float)
        rhs[indices, np.arange(len(indices))] = 1.0

        if use_gpu and self._can_use_gpu():
            system = cp.asarray(self.system_matrix)
            solution = cp.linalg.solve(system, cp.asarray(rhs))
            result = cp.asnumpy(solution)
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Device().synchronize()
            return result

        return self._solve_linear_system(rhs)

    def posterior_covariance_columns(
        self,
        indices: Iterable[int],
        *,
        use_gpu: bool = True,
    ) -> DenseArray:
        """Return selected columns of the posterior covariance matrix.

        This is often preferable to requesting the full covariance for sparse or
        large problems. The returned array has shape ``(n_model, len(indices))``.
        """

        columns = np.asarray(tuple(indices), dtype=int)
        if columns.ndim != 1:
            raise ValueError("indices must be a 1D iterable of column indices")

        size = self.system_matrix.shape[0]
        if np.any(columns < 0) or np.any(columns >= size):
            raise IndexError("posterior covariance column index out of bounds")

        if columns.size == 0:
            return np.zeros((size, 0), dtype=float)

        return self._posterior_covariance_blocks(columns, use_gpu=use_gpu)

    def posterior_covariance_diagonal(self) -> DenseArray:
        """Return the exact diagonal of the posterior covariance matrix.

        For sparse systems this solves one unit-vector right-hand side at a
        time, avoiding construction of a dense full inverse.
        """

        if self._Cmpost_diag is None:
            if self._Cmpost is not None:
                self._Cmpost_diag = np.diag(self._Cmpost).copy()
            else:
                size = self.system_matrix.shape[0]
                diagonal = np.empty(size, dtype=float)
                for idx in range(size):
                    diagonal[idx] = self.posterior_covariance_columns([idx], use_gpu=False)[idx, 0]
                self._Cmpost_diag = diagonal
        return self._Cmpost_diag.copy()

    def posterior_covariance_sparse(
        self,
        *,
        drop_tol: float = 0.0,
        batch_size: int | None = None,
        symmetrize: bool = True,
    ) -> sparse.csc_matrix:
        """Return a thresholded sparse approximation of the posterior covariance.

        The exact inverse of a sparse matrix is generally dense. This method
        computes covariance columns in batches and stores only entries with
        absolute value greater than ``drop_tol``. Use this for diagnostics, not
        as a guarantee that the posterior covariance is intrinsically sparse.
        """

        if drop_tol < 0:
            raise ValueError("drop_tol must be non-negative")

        size = self.system_matrix.shape[0]
        if batch_size is None:
            batch_size = min(size, 64) or 1
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        rows: list[DenseArray] = []
        cols: list[DenseArray] = []
        data: list[DenseArray] = []

        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            block_indices = np.arange(start, stop, dtype=int)
            block = self._posterior_covariance_blocks(block_indices, use_gpu=False)
            mask = np.abs(block) > drop_tol
            block_rows, block_cols = np.nonzero(mask)
            if block_rows.size == 0:
                continue

            rows.append(block_rows.astype(int))
            cols.append(block_indices[block_cols])
            data.append(block[block_rows, block_cols])

        if data:
            posterior = sparse.csc_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                shape=(size, size),
            )
        else:
            posterior = sparse.csc_matrix((size, size), dtype=float)

        if symmetrize:
            posterior = 0.5 * (posterior + posterior.T)
        return posterior

    @property
    def Cmpost(self) -> DenseArray:
        if self._Cmpost is None:
            size = self.system_matrix.shape[0]
            batch_size = size if not _is_sparse(self.system_matrix) else min(size, 64) or 1
            blocks = []
            for start in range(0, size, batch_size):
                stop = min(start + batch_size, size)
                block_indices = np.arange(start, stop, dtype=int)
                blocks.append(self._posterior_covariance_blocks(block_indices))
            self._Cmpost = np.hstack(blocks) if blocks else np.zeros((size, size), dtype=float)
            self._Cmpost_diag = np.diag(self._Cmpost).copy()
        return self._Cmpost

    @property
    def posterior_covariance(self) -> DenseArray:
        return self.Cmpost

    @property
    def Rm(self) -> DenseArray:
        if self._Rm is None:
            gtg = self.GTG.toarray() if _is_sparse(self.GTG) else np.asarray(self.GTG, dtype=float)
            self._Rm = self.Cmpost @ gtg
        return self._Rm

    @property
    def resolution_matrix(self) -> DenseArray:
        return self.Rm

    def get_c_factor(self) -> None:
        if _is_sparse(self.system_matrix):
            self._c_factor = None
            self._c_valid = False
            return

        try:
            self._c_factor = cho_factor(
                np.asarray(self.system_matrix, dtype=float),
                overwrite_a=False,
                check_finite=False,
                lower=True,
            )
            self._c_valid = True
        except LinAlgError:
            self._c_factor = None
            self._c_valid = False

    @property
    def c_valid(self) -> bool:
        if self._c_valid is None:
            self.get_c_factor()
        return bool(self._c_valid)

    @property
    def c_factor(self):
        if self._c_factor is None:
            self.get_c_factor()
        return self._c_factor

    def solve(self, posterior: bool = False, use_gpu: bool | None = None, **lstsq_kwargs) -> DenseArray:
        """Solve the inverse problem and return the model vector.

        Parameters
        ----------
        posterior
            If false, solve the normal equations directly. If true, compute the
            posterior covariance first and use ``Cmpost @ GTd``. The direct
            solve is usually preferable unless you already need the covariance.
        use_gpu
            Override the instance-level GPU setting for this solve.
        **lstsq_kwargs
            Extra keyword arguments for the least-squares fallback.
        """

        if use_gpu is not None:
            self.use_gpu = use_gpu

        if posterior:
            self.m = self.Cmpost @ self.GTd
        else:
            self.m = self._solve_linear_system(self.GTd, **lstsq_kwargs)
        return np.asarray(self.m, dtype=float)

    def solve_inverse_problem(self, posterior: bool = False, **kwargs) -> DenseArray:
        return self.solve(posterior=posterior, **kwargs)
