"""Higher-level interfaces for repeated inverse-problem solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy import sparse

from .solver import DenseArray, MatrixLike, QuadraticRegularization, Solver, _as_data_array, _as_matrix


def _as_regularization(
    LTL: MatrixLike,
    *,
    lreg: float = 1.0,
    scale: bool = False,
    LTL_scale: float | None = None,
) -> QuadraticRegularization:
    return QuadraticRegularization(LTL=LTL, lreg=lreg, scale=scale, LTL_scale=LTL_scale)


def _prediction(G: MatrixLike, model: DenseArray) -> DenseArray:
    return np.asarray(G @ model, dtype=float)


def _quadratic_form(model: DenseArray, matrix: MatrixLike) -> float:
    array = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
    return float(model.T @ array @ model)


def huber_weights(residual: DenseArray, delta: float, scale: float | None = None) -> DenseArray:
    """Compute IRLS weights for a Huber loss.

    Residuals smaller than ``delta * scale`` keep weight 1. Larger residuals are
    downweighted as ``delta / abs(residual / scale)``. If ``scale`` is omitted,
    a robust median absolute deviation estimate is used.
    """

    residual = np.asarray(residual, dtype=float).reshape(-1)
    if scale is None:
        median = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - median))
    scale = max(float(scale), np.finfo(float).eps)

    normalized = np.abs(residual) / scale
    weights = np.ones_like(normalized)
    mask = normalized > delta
    weights[mask] = delta / normalized[mask]
    return weights


@dataclass(frozen=True)
class SolveResult:
    """Container returned by :meth:`LinearInverseProblem.solve`.

    ``model`` is the recovered parameter vector. The remaining fields are
    simple diagnostics that are useful when comparing repeated solves.
    """

    model: DenseArray
    predicted_data: DenseArray
    residual: DenseArray
    data_misfit: float
    regularization_penalty: float
    solver: Solver


@dataclass(frozen=True)
class IRLSResult:
    """Container returned by :meth:`LinearInverseProblem.irls`.

    ``iterations`` stores every solve result so convergence can be inspected
    after the fact. ``solution`` returns the final model.
    """

    iterations: tuple[SolveResult, ...]
    converged: bool
    final_weights: DenseArray | MatrixLike | None

    @property
    def solution(self) -> DenseArray:
        return self.iterations[-1].model


class LinearInverseProblem:
    """Higher-level interface for repeated inverse-problem solves.

    This class stores a forward matrix ``G`` and data vector ``d`` and creates
    :class:`nandir.Solver` instances as needed. Use it when you want to compare
    many related solves, for example an L-curve style regularization sweep or
    an IRLS loop where the weights are updated between solves.
    """

    def __init__(
        self,
        G: object,
        d: object,
        *,
        w: object = None,
        reg: object | Iterable[object] | None = None,
        **solver_kwargs,
    ) -> None:
        self.G = _as_matrix("G", G)
        self.d = _as_data_array("d", d)
        self.default_w = w
        self.default_reg = reg
        self.solver_kwargs = solver_kwargs

        if self.d is None:
            raise ValueError("d must be provided")
        if self.G.shape[0] != self.d.shape[0]:
            raise ValueError("G and d must have the same number of rows")

    def solver(self, *, w: object = None, reg: object | Iterable[object] | None = None, **kwargs) -> Solver:
        """Create a low-level :class:`Solver` using this problem's ``G`` and ``d``."""

        merged_kwargs = self.solver_kwargs | kwargs
        return Solver(
            G=self.G,
            d=self.d,
            w=self.default_w if w is None else w,
            reg=self.default_reg if reg is None else reg,
            **merged_kwargs,
        )

    def normal_equations(self, *, w: object = None) -> tuple[MatrixLike, DenseArray]:
        """Return ``GTG`` and ``GTd`` for the current data weighting.

        This is useful when only regularization changes across many solves,
        because ``GTG`` and ``GTd`` can be reused.
        """

        solver = self.solver(w=w, reg=None)
        return solver.GTG, solver.GTd

    def solver_from_normal_equations(
        self,
        *,
        GTG: MatrixLike,
        GTd: DenseArray,
        reg: object | Iterable[object] | None = None,
        **kwargs,
    ) -> Solver:
        """Create a :class:`Solver` from precomputed ``GTG`` and ``GTd``."""

        merged_kwargs = self.solver_kwargs | kwargs
        return Solver(
            GTG=GTG,
            GTd=GTd,
            reg=self.default_reg if reg is None else reg,
            **merged_kwargs,
        )

    def solve(
        self,
        *,
        w: object = None,
        reg: object | Iterable[object] | None = None,
        posterior: bool = False,
        **kwargs,
    ) -> SolveResult:
        """Solve the inverse problem and return the model plus diagnostics."""

        solver = self.solver(w=w, reg=reg, **kwargs)
        model = solver.solve(posterior=posterior)
        predicted = _prediction(self.G, model)
        residual = np.asarray(self.d - predicted, dtype=float)
        misfit = float(np.linalg.norm(residual))
        penalty = _quadratic_form(model, solver.LTL)
        return SolveResult(
            model=model,
            predicted_data=predicted,
            residual=residual,
            data_misfit=misfit,
            regularization_penalty=penalty,
            solver=solver,
        )

    def solve_many(self, configs: Iterable[dict]) -> list[SolveResult]:
        """Run a list of solve configurations.

        Each item in ``configs`` is passed as keyword arguments to ``solve``.
        """

        return [self.solve(**config) for config in configs]

    def regularization_path(
        self,
        lambdas: Iterable[float],
        LTL: MatrixLike,
        *,
        scale: bool = False,
        w: object = None,
        posterior: bool = False,
        **kwargs,
    ) -> list[SolveResult]:
        """Solve the same problem for several regularization strengths.

        The normal equations are computed once and reused for every value in
        ``lambdas``. This is the basic building block for L-curve style
        experiments.
        """

        GTG, GTd = self.normal_equations(w=w)
        results = []
        for lreg in lambdas:
            reg = _as_regularization(LTL, lreg=float(lreg), scale=scale)
            solver = self.solver_from_normal_equations(GTG=GTG, GTd=GTd, reg=reg, **kwargs)
            model = solver.solve(posterior=posterior)
            predicted = _prediction(self.G, model)
            residual = np.asarray(self.d - predicted, dtype=float)
            results.append(
                SolveResult(
                    model=model,
                    predicted_data=predicted,
                    residual=residual,
                    data_misfit=float(np.linalg.norm(residual)),
                    regularization_penalty=_quadratic_form(model, solver.LTL),
                    solver=solver,
                )
            )
        return results

    def irls(
        self,
        update_weights: Callable[[SolveResult, int], object],
        *,
        initial_w: object = None,
        reg: object | Iterable[object] | None = None,
        max_iterations: int = 25,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        posterior: bool = False,
        **kwargs,
    ) -> IRLSResult:
        """Run an iteratively reweighted least-squares loop.

        ``update_weights`` is called as ``update_weights(result, iteration)``
        after each solve and must return the weights for the next iteration.
        """

        weights = self.default_w if initial_w is None else initial_w
        previous_model = None
        iterations: list[SolveResult] = []
        converged = False

        for iteration in range(max_iterations):
            result = self.solve(w=weights, reg=reg, posterior=posterior, **kwargs)
            iterations.append(result)

            if previous_model is not None:
                delta = np.linalg.norm(result.model - previous_model)
                baseline = max(np.linalg.norm(previous_model), 1.0)
                if delta <= atol + rtol * baseline:
                    converged = True
                    break

            previous_model = result.model
            weights = update_weights(result, iteration)

        return IRLSResult(
            iterations=tuple(iterations),
            converged=converged,
            final_weights=weights,
        )

    def huber_irls(
        self,
        delta: float,
        *,
        initial_w: object = None,
        reg: object | Iterable[object] | None = None,
        scale: float | None = None,
        max_iterations: int = 25,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        posterior: bool = False,
        **kwargs,
    ) -> IRLSResult:
        """Run IRLS with Huber weights computed from the residuals."""

        def update(result: SolveResult, _iteration: int) -> DenseArray:
            return huber_weights(result.residual, delta=delta, scale=scale)

        return self.irls(
            update,
            initial_w=initial_w,
            reg=reg,
            max_iterations=max_iterations,
            rtol=rtol,
            atol=atol,
            posterior=posterior,
            **kwargs,
        )
