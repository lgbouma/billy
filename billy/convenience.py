def chisq(y_mod, y_obs, y_err):
    return np.sum( (y_mod - y_obs )**2 / y_err**2 )

def bic(chisq, k, n):
    """
    BIC = χ^2 + k log n, for k the number of free parameters, and n the
    number of data points.
    """
    return chisq + k*np.log(n)


