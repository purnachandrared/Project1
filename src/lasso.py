import numpy as np
from scipy import linalg

class LassoRegression:
    """
    LASSO regression implementation using the Homotopy Method.
    
    The Homotopy Method efficiently computes the LASSO solution path by
    starting with a large regularization parameter and gradually decreasing it.
    """
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        """
        Initialize the LASSO regression model.
        
        Parameters:
        -----------
        alpha : float, optional (default=1.0)
            Regularization parameter. Larger values create more sparsity.
        max_iter : int, optional (default=1000)
            Maximum number of iterations for the homotopy method.
        tol : float, optional (default=1e-4)
            Convergence tolerance.
        """
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
            
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0
        
    def fit(self, X, y):
        """
        Fit the LASSO model using the Homotopy Method.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if y.size == 0:
            raise ValueError("y cannot be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
            
        # Center the data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        n_samples, n_features = X_centered.shape
        
        # Initialize coefficients
        beta = np.zeros(n_features)
        active_set = set()
        
        # Compute initial correlations
        correlations = X_centered.T @ y_centered
        max_corr = np.max(np.abs(correlations))
        
        # Start with large lambda and decrease
        lambda_max = max_corr
        lambda_current = lambda_max
        
        for _ in range(self.max_iter):
            # Find the feature with maximum correlation
            corr_abs = np.abs(correlations)
            max_idx = np.argmax(corr_abs)
            
            if corr_abs[max_idx] < self.tol:
                break
                
            # Update active set
            if max_idx not in active_set:
                active_set.add(max_idx)
            
            # Compute the direction of movement
            X_active = X_centered[:, list(active_set)]
            if len(active_set) == 1:
                direction = np.sign(correlations[max_idx])
            else:
                # Solve the system of equations
                try:
                    direction = linalg.solve(X_active.T @ X_active,
                                           np.sign(correlations[list(active_set)]))
                except linalg.LinAlgError:
                    break
            
            # Find the step size
            gamma = np.inf
            for j in range(n_features):
                if j not in active_set:
                    r_j = correlations[j]
                    d_j = X_centered[:, j] @ X_active @ direction
                    if abs(d_j) > self.tol:
                        gamma_j = (lambda_current - r_j) / (1 - d_j)
                        if gamma_j > 0 and gamma_j < gamma:
                            gamma = gamma_j
                            drop_idx = j
            
            # Update coefficients
            beta[list(active_set)] += gamma * direction
            
            # Update correlations
            correlations -= gamma * (X_centered.T @ (X_active @ direction))
            
            # Check for coefficient dropping
            if gamma < lambda_current:
                lambda_current -= gamma
            else:
                lambda_current = 0
                
            # Remove variables from active set if their coefficients become zero
            active_set = {j for j in active_set if abs(beta[j]) > self.tol}
            
            if lambda_current < self.alpha:
                break
        
        self.coef_ = beta
        self.intercept_ = y_mean - np.dot(X_mean, beta)
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        C : array-like of shape (n_samples,)
            Returns predicted values.
        """
        X = np.asarray(X)
        return np.dot(X, self.coef_) + self.intercept_
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters:
        -----------
        **params : dict
            Estimator parameters.
            
        Returns:
        --------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self 