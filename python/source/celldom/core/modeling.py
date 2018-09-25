import numpy as np
from sklearn.linear_model import LinearRegression


def get_growth_rate(time, value, e=1):
    if len(time) != len(value) or time.ndim > 1 or value.ndim > 1:
        raise ValueError(
            'Time and value should be 1D with same length (shape time = {}, shape value = {})'
            .format(time.shape, value.shape)
        )
    if np.any(value < 0):
        raise ValueError('All values must be >= 0 (values given = {})'.format(value))

    n = len(time)
    if n < 2:
        return np.nan

    # Run log linear regression and return slope coefficient
    X = np.expand_dims(time, -1)
    y = np.log2(value + e)
    m = LinearRegression().fit(X, y)
    return m.coef_[0]
