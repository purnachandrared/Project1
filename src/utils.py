import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n_samples=100, n_features=10, sparsity=0.3, noise=0.1):
    """
    Generate synthetic data for testing LASSO regression.
    
    Parameters:
    -----------
    n_samples : int, optional (default=100)
        Number of samples.
    n_features : int, optional (default=10)
        Number of features.
    sparsity : float, optional (default=0.3)
        Proportion of non-zero coefficients.
    noise : float, optional (default=0.1)
        Standard deviation of noise added to target.
        
    Returns:
    --------
    X : array-like of shape (n_samples, n_features)
        Generated features.
    y : array-like of shape (n_samples,)
        Generated target values.
    true_coef : array-like of shape (n_features,)
        True coefficients used to generate the data.
    """
    np.random.seed(42)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate sparse coefficients
    n_nonzero = int(n_features * sparsity)
    true_coef = np.zeros(n_features)
    true_coef[:n_nonzero] = np.random.randn(n_nonzero)
    np.random.shuffle(true_coef)
    
    # Generate target values with noise
    y = np.dot(X, true_coef) + np.random.randn(n_samples) * noise
    
    return X, y, true_coef

def plot_lasso_path(X, y, alphas=np.logspace(-3, 3, 100)):
    """
    Plot the LASSO solution path.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.
    alphas : array-like, optional
        Array of alpha values to try.
    """
    from src.lasso import LassoRegression
    
    coefs = []
    for alpha in alphas:
        model = LassoRegression(alpha=alpha)
        model.fit(X, y)
        coefs.append(model.coef_)
    
    coefs = np.array(coefs)
    
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], label=f'Feature {i+1}')
    
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.title('LASSO Solution Path')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : LassoRegression
        Fitted LASSO model.
    X_test : array-like of shape (n_samples, n_features)
        Test features.
    y_test : array-like of shape (n_samples,)
        Test target values.
        
    Returns:
    --------
    dict
        Dictionary containing various performance metrics.
    """
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    
    return {
        'mse': mse,
        'r2': r2,
        'n_nonzero_features': n_nonzero
    }

def generate_collinear_data(n_samples=100, n_features=10, noise=0.1):
    """
    Generate data with collinear features for testing LASSO's feature selection.
    
    Parameters:
    -----------
    n_samples : int, optional (default=100)
        Number of samples.
    n_features : int, optional (default=10)
        Number of features.
    noise : float, optional (default=0.1)
        Standard deviation of noise added to target.
        
    Returns:
    --------
    X : array-like of shape (n_samples, n_features)
        Generated features with collinear columns.
    y : array-like of shape (n_samples,)
        Generated target values.
    """
    np.random.seed(42)
    
    # Generate base features
    X_base = np.random.randn(n_samples, n_features // 2)
    
    # Create collinear features
    X_collinear = X_base + np.random.randn(n_samples, n_features // 2) * 0.1
    
    # Combine features
    X = np.hstack([X_base, X_collinear])
    
    # Generate target values
    true_coef = np.random.randn(n_features)
    y = np.dot(X, true_coef) + np.random.randn(n_samples) * noise
    
    return X, y 