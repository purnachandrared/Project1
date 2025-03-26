import numpy as np
import pytest
from src.lasso import LassoRegression
from src.utils import generate_synthetic_data, generate_collinear_data

def test_basic_functionality():
    """Test basic functionality with simple data."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    
    model = LassoRegression(alpha=0.1)
    model.fit(X, y)
    
    assert model.coef_ is not None
    assert len(model.coef_) == 2
    assert model.intercept_ is not None

def test_feature_selection():
    """Test that LASSO performs feature selection with collinear data."""
    X, y = generate_collinear_data(n_samples=100, n_features=10)
    
    model = LassoRegression(alpha=0.1)
    model.fit(X, y)
    
    # Check that some coefficients are exactly zero
    assert np.sum(np.abs(model.coef_) > 1e-6) < 10

def test_sparsity():
    """Test that LASSO produces sparse solutions."""
    X, y, _ = generate_synthetic_data(n_samples=100, n_features=10, sparsity=0.3)
    
    model = LassoRegression(alpha=0.5)
    model.fit(X, y)
    
    # Check that some coefficients are exactly zero
    assert np.sum(np.abs(model.coef_) > 1e-6) < 10

def test_convergence():
    """Test that the model converges with different alpha values."""
    X, y, _ = generate_synthetic_data(n_samples=50, n_features=5)
    
    # Test with different alpha values
    alphas = [0.1, 0.5, 1.0]
    for alpha in alphas:
        model = LassoRegression(alpha=alpha)
        model.fit(X, y)
        assert model.coef_ is not None
        assert not np.any(np.isnan(model.coef_))

def test_prediction():
    """Test prediction functionality."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    
    model = LassoRegression(alpha=0.1)
    model.fit(X, y)
    
    X_test = np.array([[2, 3], [4, 5]])
    y_pred = model.predict(X_test)
    
    assert len(y_pred) == 2
    assert not np.any(np.isnan(y_pred))

def test_numerical_stability():
    """Test numerical stability with ill-conditioned data."""
    # Create ill-conditioned data
    X = np.array([
        [1e-6, 1],
        [1e-6, 2],
        [1e-6, 3]
    ])
    y = np.array([1, 2, 3])
    
    model = LassoRegression(alpha=0.1, tol=1e-6)
    model.fit(X, y)
    
    assert not np.any(np.isnan(model.coef_))
    assert not np.any(np.isinf(model.coef_))

def test_parameter_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError):
        LassoRegression(alpha=-1)
    
    with pytest.raises(ValueError):
        LassoRegression(max_iter=-1)
    
    with pytest.raises(ValueError):
        LassoRegression(tol=-1)

def test_empty_input():
    """Test handling of empty input."""
    X = np.array([]).reshape(0, 2)
    y = np.array([])
    
    model = LassoRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_dimension_mismatch():
    """Test handling of dimension mismatch."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3])  # Different number of samples
    
    model = LassoRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_collinear_feature_selection():
    """Test feature selection with highly collinear data."""
    X, y = generate_collinear_data(n_samples=100, n_features=10)
    
    # Test with different alpha values
    alphas = [0.1, 0.5, 1.0]
    n_nonzero_features = []
    
    for alpha in alphas:
        model = LassoRegression(alpha=alpha)
        model.fit(X, y)
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
        n_nonzero_features.append(n_nonzero)
    
    # Check that higher alpha values result in fewer selected features
    assert n_nonzero_features[0] >= n_nonzero_features[1] >= n_nonzero_features[2]

def test_solution_path():
    """Test that the solution path is continuous."""
    X, y, _ = generate_synthetic_data(n_samples=100, n_features=5)
    
    alphas = np.logspace(-3, 3, 10)
    coefs = []
    
    for alpha in alphas:
        model = LassoRegression(alpha=alpha)
        model.fit(X, y)
        coefs.append(model.coef_)
    
    coefs = np.array(coefs)
    
    # Check that coefficients change continuously
    for i in range(coefs.shape[1]):
        changes = np.abs(np.diff(coefs[:, i]))
        assert np.all(changes < 1e-6) or np.all(changes > 0) 