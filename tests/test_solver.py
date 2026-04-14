import numpy as np
from scipy import sparse

from nandir import LinearInverseProblem, QuadraticRegularization, Solver, huber_weights


def test_diagonal_vector_weight_matches_dense_weight():
    G = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]])
    d = np.array([1.0, -2.0, 0.5])
    w = np.array([2.0, 0.5, 3.0])

    vector_solver = Solver(G=G, d=d, w=w)
    dense_solver = Solver(G=G, d=d, w=np.diag(w))

    vector_model = vector_solver.solve()
    dense_model = dense_solver.solve()

    np.testing.assert_allclose(vector_solver.GTG, dense_solver.GTG)
    np.testing.assert_allclose(vector_solver.GTd, dense_solver.GTd)
    np.testing.assert_allclose(vector_model, dense_model)


def test_posterior_covariance_uses_regularized_system():
    G = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    d = np.array([1.0, 2.0, 1.0])
    reg = QuadraticRegularization(np.eye(2), lreg=0.25)

    solver = Solver(G=G, d=d, reg=reg)

    expected = np.linalg.inv(G.T @ G + 0.25 * np.eye(2))
    np.testing.assert_allclose(solver.posterior_covariance, expected)


def test_precomputed_normal_equations_are_supported():
    GTG = np.array([[5.0, 1.0], [1.0, 4.0]])
    GTd = np.array([3.0, 2.0])

    solver = Solver(GTG=GTG, GTd=GTd)

    np.testing.assert_allclose(solver.solve(), np.linalg.solve(GTG, GTd))


def test_scaled_regularization_matches_manual_system():
    G = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    d = np.array([1.0, -1.0, 0.5])
    LTL = np.array([[2.0, 0.0], [0.0, 8.0]])
    reg = QuadraticRegularization(LTL=LTL, lreg=0.5, scale=True)

    solver = Solver(G=G, d=d, reg=reg)

    gtg = G.T @ G
    scaled_ltl = 0.5 * LTL / np.median(np.diag(LTL)) * np.median(np.diag(gtg))
    expected = np.linalg.solve(gtg + scaled_ltl, G.T @ d)

    np.testing.assert_allclose(solver.solve(), expected)


def test_sparse_forward_matrix_matches_dense_solution():
    G = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]])
    d = np.array([1.0, -2.0, 0.5])
    w = np.array([2.0, 0.5, 3.0])

    dense_solver = Solver(G=G, d=d, w=w)
    sparse_solver = Solver(G=sparse.csr_matrix(G), d=d, w=w)

    np.testing.assert_allclose(sparse_solver.solve(), dense_solver.solve())
    np.testing.assert_allclose(sparse_solver.GTd, dense_solver.GTd)
    np.testing.assert_allclose(sparse_solver.GTG.toarray(), dense_solver.GTG)


def test_sparse_weight_matrix_is_supported():
    G = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    d = np.array([1.0, 2.0, 1.0])
    w = sparse.diags([1.0, 2.0, 1.0])

    solver = Solver(G=G, d=d, w=w)
    expected = np.linalg.solve((G.T @ w @ G).toarray(), np.asarray(G.T @ (w @ d)).reshape(-1))

    np.testing.assert_allclose(solver.solve(), expected)


def test_sparse_posterior_covariance_columns_match_dense_inverse():
    GTG = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    GTd = np.array([1.0, 2.0])
    solver = Solver(GTG=GTG, GTd=GTd)

    expected = np.linalg.inv(GTG.toarray())
    np.testing.assert_allclose(solver.posterior_covariance_columns([0, 1]), expected)


def test_sparse_posterior_covariance_diagonal_matches_dense_inverse():
    GTG = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    GTd = np.array([1.0, 2.0])
    solver = Solver(GTG=GTG, GTd=GTd)

    expected = np.diag(np.linalg.inv(GTG.toarray()))
    np.testing.assert_allclose(solver.posterior_covariance_diagonal(), expected)


def test_sparse_posterior_covariance_sparse_returns_sparse_matrix():
    GTG = sparse.diags([2.0, 4.0, 8.0], format="csr")
    GTd = np.array([1.0, 1.0, 1.0])
    solver = Solver(GTG=GTG, GTd=GTd)

    posterior = solver.posterior_covariance_sparse(drop_tol=1e-14)

    assert sparse.issparse(posterior)
    np.testing.assert_allclose(posterior.toarray(), np.diag([0.5, 0.25, 0.125]))


def test_regularization_path_returns_multiple_results():
    G = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    d = np.array([1.0, 2.0, 1.0])
    problem = LinearInverseProblem(G, d)

    results = problem.regularization_path([0.0, 0.5, 1.0], np.eye(2))

    assert len(results) == 3
    assert results[0].data_misfit <= results[-1].data_misfit


def test_huber_weights_downweight_outlier():
    residual = np.array([0.1, 0.2, 10.0])
    weights = huber_weights(residual, delta=1.5, scale=1.0)

    np.testing.assert_allclose(weights[:2], np.ones(2))
    assert weights[2] < 1.0


def test_huber_irls_runs_multiple_solves():
    G = np.array([[1.0], [1.0], [1.0]])
    d = np.array([1.0, 1.2, 10.0])
    problem = LinearInverseProblem(G, d)

    result = problem.huber_irls(delta=1.5, max_iterations=5, scale=1.0)

    assert len(result.iterations) >= 2
    assert result.solution.shape == (1,)
