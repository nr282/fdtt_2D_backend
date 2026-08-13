"""
Module handles verification of the 2D Grid.

"""

from typing import Callable
import numpy as np
from scipy.integrate import dblquad


def manual_d2f_dx2(func, x, y, h=1e-4):
    return (func(x + h, y) - 2 * func(x, y) + func(x - h, y)) / (h**2)

def manual_d2f_dy2(func, x, y, h=1e-4):
    return (func(x, y+h) - 2 * func(x, y) + func(x, y-h)) / (h**2)

def verification_of_laplacian(func: Callable[[float, float], float],
                              X: float,
                              Y: float,
                              h: float,
                              lambdas: np.ndarray = None,
                              f = lambda x,y: 0,
                              grid: np.ndarray = None,
                              M: int = None,
                              N: int = None,
                              check_del = False,
                              ):
    """
    Verification of the func Callable.

    TODO: We need to ensure that the integrals line up.
    TODO:

    :param fun:
    :return:
    """

    if check_del:
        m,n = X//h, Y//h
        epsilon = 0.1
        del_calcs = []
        for i in range(10):
            x, y = np.random.uniform(low=0, high=X), np.random.uniform(low=0, high=Y)
            del_calc = manual_d2f_dx2(func, x, y) + manual_d2f_dy2(func, x, y)
            if lambdas is not None:
                lambda_val = lambdas[int(x//h) * m + int(y//h)]
                if abs(abs(del_calc) - lambda_val) > epsilon:
                    print(f"Solver has failed. The del calc is {del_calc} "
                          f"and the lambdas is {lambda_val}")
            else:
                delta = abs(abs(del_calc) - f(x,y))
                print("Checking Lapalacian is close to specfiied value")
                if  delta > epsilon:
                    raise RuntimeError(f"Solver has failed. The del calc is {del_calc}")

    if grid is not None:
        epsilon = 0.01
        grid_m, grid_n = grid.shape
        #Check the integrals
        for i in range(grid_m):
            for j in range(grid_n):
                print(f"Grid Cell I: {i} and J: {j}")
                actual_integral_value = grid[i,j]
                calculated_integral,_ = dblquad(func, i*h, (i+1)*h, j*h, (j+1)*h, epsrel=1e-3, epsabs=1e-3)
                assert(abs(calculated_integral - actual_integral_value) < epsilon,
                       f"Calculated Integral {calculated_integral} does not match actual integral "
                       f"{actual_integral_value}")

        return True


