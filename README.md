# LASSO Regression with Homotopy Method

This project implements the LASSO (Least Absolute Shrinkage and Selection Operator) regression model using the Homotopy Method. The implementation is done from first principles, without using built-in models from scikit-learn.

## What is LASSO Regression?

LASSO regression is a type of linear regression that performs both variable selection and regularization. It adds an L1 penalty term to the ordinary least squares objective function, which helps in:
- Feature selection by setting some coefficients to exactly zero
- Handling multicollinearity in the data
- Preventing overfitting

The Homotopy Method is an efficient algorithm for computing the LASSO solution path as the regularization parameter varies from infinity to zero.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── lasso.py          # Main LASSO implementation
│   └── utils.py          # Utility functions
├── tests/
│   ├── __init__.py
│   └── test_lasso.py     # Test cases
├── notebooks/
│   └── examples.ipynb    # Example usage and visualizations
├── requirements.txt
└── README.md
```

## Usage

```python
from src.lasso import LassoRegression

# Create and fit the model
model = LassoRegression(alpha=0.1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Model Parameters

- `alpha`: Regularization parameter (default=1.0)
  - Larger values create more sparsity
  - Smaller values allow more features to be selected
- `max_iter`: Maximum number of iterations for the homotopy method (default=1000)
- `tol`: Convergence tolerance (default=1e-4)

## When to Use This Model

This LASSO implementation is particularly useful when:
1. You have high-dimensional data with many features
2. You suspect some features are irrelevant or redundant
3. You want to perform feature selection while fitting the model
4. Your data exhibits multicollinearity

## Testing Methodology

The model has been tested for:
1. Correctness of the homotopy method implementation
2. Feature selection behavior with collinear data
3. Convergence properties
4. Numerical stability
5. Comparison with theoretical expectations

## Known Limitations

1. The current implementation may be slower than optimized libraries for very large datasets
2. Some edge cases with extremely ill-conditioned data may cause numerical instability
3. The homotopy method may require more iterations for certain data configurations

## Future Improvements

1. Implement parallel processing for faster computation
2. Add support for different loss functions
3. Implement cross-validation for automatic alpha selection
4. Add more robust numerical stability checks
5. Implement early stopping criteria

## License

MIT License 