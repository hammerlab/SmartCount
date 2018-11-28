import numpy as np
from sklearn.linear_model import LinearRegression


def get_growth_rate(time, value, e=1, fit_intercept=True):
    if len(time) != len(value) or time.ndim > 1 or value.ndim > 1:
        raise ValueError(
            'Time and value should be 1D with same length (shape time = {}, shape value = {})'
            .format(time.shape, value.shape)
        )
    if np.any(value < 0):
        raise ValueError('All values must be >= 0 (values given = {})'.format(value))

    n = len(time)
    if n < 2:
        return np.nan, np.nan

    # Run log linear regression and return slope coefficient
    X = np.expand_dims(time, -1)
    y = np.log2(value + e)
    m = LinearRegression(fit_intercept=fit_intercept).fit(X, y)

    # Intercept is 0 when fit_intercept False
    return m.intercept_, m.coef_[0]
