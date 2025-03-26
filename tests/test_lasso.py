import numpy as np
import pytest
from src.lasso import LassoRegression

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
    # Create collinear features
    X = np.array([
        [1, 1, 0],  # First two features are identical
        [2, 2, 1],
        [3, 3, 2]
    ])
    y = np.array([1, 2, 3])
    
    model = LassoRegression(alpha=0.1)
    model.fit(X, y)
    
    # Check that one of the collinear features is set to zero
    assert np.sum(np.abs(model.coef_) > 1e-6) < 3

def test_sparsity():
    """Test that LASSO produces sparse solutions."""
    # Create data with some irrelevant features
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.dot(X[:, :3], [1, 2, 3]) + np.random.randn(100) * 0.1
    
    model = LassoRegression(alpha=0.5)
    model.fit(X, y)
    
    # Check that some coefficients are exactly zero
    assert np.sum(np.abs(model.coef_) > 1e-6) < 10

def test_convergence():
    """Test that the model converges with different alpha values."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.dot(X, [1, 2, 0, 0, 0]) + np.random.randn(50) * 0.1
    
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