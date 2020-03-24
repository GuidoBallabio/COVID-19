import numpy as np
from scipy.optimize import curve_fit

class Fitter:
    def __init__(self, *params):
        self.params = params
        self.sigma = np.zeros_like(params)

    @classmethod
    def fit(cls, x, y, **kwargs):
        est, cov = curve_fit(cls._compute, x, y, **kwargs)
        fitter = cls(*est)
        fitter.sigmas = np.sqrt(np.diag(cov))
        return fitter

    def compute(self, x, *params):
        if not params:
            params = self.params
        return self.__class__._compute(x, *params)

    def compute_inverse(self, x, *params):
        if not params:
            params = self.params
        return self.__class__._compute_inverse(x, *params)

    def compute_derivative(self, x, *params):
        if not params:
            params = self.params
        return self.__class__._compute_derivative(x, *params)
    
    def plateau(self, *params):
        if not params:
            params = self.params
        return self.__class__._plateau(*params)

    def peak(self, *params):
        if not params:
            params = self.params
        return self.__class__._peak(*params)

    def inverse_perc(self, perc, *params):
        return self.compute_inverse(perc * self.plateau(), *params)

    def __str__(self):
        return str(self.__class__.__name__) + f"{self.params}"
    
    @classmethod
    def _compute(cls, x, *params):
        raise NotImplemented

    @classmethod
    def _compute_inverse(cls, x, *params):
        raise NotImplemented

    @classmethod
    def _compute_derivative(cls, x, *params):
        raise NotImplemented

    @classmethod
    def _plateau(cls, *params):
        raise NotImplemented

    @classmethod
    def _peak(cls, *params):
        raise NotImplemented


class Logistic(Fitter):
    def __init__(self, x_0, k, L):
        super().__init__(*(x_0, k, L))
    
    @classmethod
    def _compute(cls, x, x_0, k, L):
        return L/(1 +  np.exp(-k*(x-x_0)))

    @classmethod
    def _compute_inverse(cls, y, x_0, k, L):
        if y <= 0:
            return -np.inf
        return -np.log(L/y - 1)/k + x_0

    @classmethod
    def _compute_derivative(cls, x, x_0, k, L):
        r = cls._compute(x, x_0, k, 1)
        return k * L * r * (1 - r)

    @classmethod
    def _plateau(cls, x_0, k, L):
        return L
    
    @classmethod
    def _peak(cls, x_0, k, L):
        return x_0


class Gompertz(Fitter):
    def __init__(self, y_0, k, L):
        super().__init__(*(y_0, k, L))
    
    @classmethod
    def _compute(cls, x, y_0, k, L):
        return L * np.exp(- np.log(L/y_0) * np.exp(-k * x))

    @classmethod
    def _compute_inverse(cls, y, y_0, k, L):
        if y <= 0:
            return -np.inf
        return - np.log( np.log(L / y) / np.log(L / y_0) ) / k

    @classmethod
    def _compute_derivative(cls, x, y_0, k, L):
        return L * np.log(L/y_0) * k * np.exp(np.log(L/y_0) * np.exp(- k * x) - k * x)
    
    @classmethod
    def _plateau(cls, y_0, k, L):
        return L
    
    @classmethod
    def _peak(cls, y_0, k, L):
        return np.log(np.log(L/y_0)) / k


class GenLogistic(Fitter):
    def __init__(self, x_0, k, L, m):
        super().__init__(*(x_0, k, L, m))
    
    @classmethod
    def _compute(cls, x, x_0, k, L, m):
        return L * (1 +  np.exp(-k*(x-x_0)))**(-1/m)

    @classmethod
    def _compute_inverse(cls, y, x_0, k, L, m):
        if y <= 0:
            return -np.inf
        return -np.log((L/y)**m - 1)/k + x_0

    @classmethod
    def _compute_derivative(cls, x, x_0, k, L, m):
        r = cls._compute(x, x_0, k, 1, m)
        return - L * k * np.exp(k*(x-x_0)) * r**(m + 1) / m

    @classmethod
    def _plateau(cls, x_0, k, L, m):
        return L
    
    @classmethod
    def _peak(cls, x_0, k, L, m):
        return cls._compute_inverse(L/2, x_0, k, L, m)