"""
Module handles the 2D version of FDTT, using Calculus Of Variations.

Core Ideas required for the module are:
    1. Calculus of Variations in 2D
        - Math generalizes from the 1D Case for FDTT.
    2. Calculus of Variations produces a laplacian operator and a right hand side
        - This is the Poisson Equation with a function f in the grid.
        - (d^2 u / d^2 x) + (d^2 u / d^2 y) u = f.
        - u is the solution
        - f is the forcing function provided by the encoding of the integral constraints.
    3. Solution to the Poisson Equation with a function f in the grid is solved by decomposing
    f into sines and cosines.
    4. f is the linear combination of indicator functions.
    5. This requires the approximation of an indicator function with a set of sine and cosines.
    6. Such a construction will be inhibited by the Gibbs Phenomena, which states that convergence
    cannot be uniform, but only pointwise, so this is an issue that we will need to deal with.

Other implementation ideas:
     1. Take in a grid representing pixels
        - This will include:
            + h, the size of the pixel.
            + m, the number of sines in x-axis
            + n, the number of cosines in y-axis
    2. Calculate the matrix to solve.
    3. solve matrix, save matrix, LU Decompose Matrix.
    4. return lambda taking two parameters representing the grid.
    5. do verification of the grid if necessary.


There was a new formulation that was developed.
    1. The mathematics has been changed such that there may be issues around the
    edges of the picture because there are Dirichlet boundary conditions.
    2. There is now a new mathematical formulation:
        1. First, I solve the Dirichlet problem on the 2D grid.
            - This can be formulated as a solution to the problem.
            - This can then also be formulated as a problem.
        2. Second, I solve the Calculus of Variations problem with zero boundary with
        a particular forcing function for the right hand side as a modified problem
        from the first problem.


------------------------------------------------------------------------------------------
A diagram mapping the steps can be very helpful.

What is the full problem we look to solve. Can we break this problem down into components?
    1. After solving each component the code can be abstracted?

    Main Problem:
        1. Minimize Energy
        2. Subject to Certain:
            a. Integral Constraints
            b. Boundary Conditions


    The solution is to decompose the problem.
        1. Solve the Laplace Equation with a particular condition.
            - In our case, we solve it with the average values on the boundary.
        2. Reformulate the Calculus of Variations problem from P into Q.
        2. Place Q into the Calculus of Variations solver with Dirichlet Boundary Condition.
        3. Solve the problem with Q.
        4. Add the solutions together. Solution to P and the Solution to Q, (ie, P + Q).


The code solver can the be broken down into a few components:
    1. Solve Dirichlet Problem
    2. Integrate Dirichlet Problem
    3. Created modified problem.
    4. Pass the modified problem to the solver.

In a diagram:
                                Initial Problem
                                        |
                                Solve Dirichlet
                                        |
                                Integrate Dirichlet Problem
                                        |
                        Pass new constraints to Calculus of Variations
                                        |
                            Solve the new Problem
                                        |
                            Add the solutions together
                                        |

How should I represent the grid?
    1. From our observations of imageGrid, it is represented as an array
    of the following shape: (857, 1280, 3).
    2. Each pixel is uniform in its shape (100,100,100)
    3. If we want to in the future, extend this to a new problem, it might be better to implement
    with a rectangular grid.
    4. Further, it might be able to abstract certain portions of the application for further extension.
        1. Can we abstract the Dirichlet Solver


TODO:
    TODO: Double Check that Math for Errors
    TODO: Ensure that there are no issues and that the result lines
    TODO: up perfectly.

"""

from typing import Callable, List
import numpy as np
import math
from collections.abc import Callable
from typing import Protocol
from functools import partial
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
import numpy as np
import logging
from calculus_of_variations_solver import create_two_dimensional_function
from enum import Enum
from verification import manual_d2f_dy2, manual_d2f_dx2, verification_of_laplacian

logging.basicConfig(level=logging.DEBUG)


USE_ANALYTICAL = True


class DIRECTION(Enum):
    vertical = 0
    horizontal = 1


class EDGE(Enum):
    low = 0
    high = 1


class Boundary(object):
    """
    A boundary represents the edges of the grid.

    For instance, if the grid is provided by the domain:
        grid = [0,M] x [0,N]

    There are four boundaries represented below:

        -------------------  (high_h)
        |                  |
        |                  |
        |                  |
        |                  |
        |                  |
        -------------------  (low_h)
    left_v              right_v
    """

    def __init__(self,
                 low_h,
                 high_h,
                 left_v,
                 right_v):

        self.low_h = low_h
        self.high_h = high_h
        self.left_v = left_v
        self.right_v = right_v



def generate_grid(n, x=None,y=None) -> np.ndarray:

    if x is None or y is None:
        grid = np.random.rand(n,n)
        return grid
    else:
        grid = np.random.rand(x,y)
        return grid

def generate_zero_grid(n) -> np.ndarray:

    grid = np.zeros((n,n))
    return grid

class LaplaceProblem(object):

    def __init__(self, boundary: Boundary, h: float, X: float, Y: float, grid):
        self.boundary = boundary
        self.h = h
        self.X = X
        self.Y = Y
        self.grid = grid


class BoundaryManager(object):

    def __init__(self, f, direction: DIRECTION, edge: EDGE, grid, h: float):
        m, n = grid.shape
        self.m = m
        self.n = n
        self.f = f
        self.h = h
        if direction == DIRECTION.horizontal and edge == EDGE.low:
            self.one_d_grid = [grid[0,i] for i in range(n)]
        elif direction == DIRECTION.horizontal and edge == EDGE.high:
            self.one_d_grid = [grid[m-1,i] for i in range(n)]
        elif direction == DIRECTION.vertical and edge == EDGE.low:
            self.one_d_grid = [grid[i,0] for i in range(m)]
        elif direction == DIRECTION.vertical and edge == EDGE.high:
            self.one_d_grid = [grid[i,n-1] for i in range(m)]


    def __call__(self, x):
        return self.f(x)


def low_h(grid, h: float, x: int):

    m,n=grid.shape

    j = int(x//h)
    if j < n:
        return grid[0, j]
    else:
        raise RuntimeError(f"j {j} is out of bounds for n {n}")

def high_h(grid, h_step, x):

    m, n = grid.shape
    j = int(x // h_step)
    if j < n:
        return grid[m-1,j]
    else:
        raise RuntimeError(f"j {j} is out of bounds for n {n}")

def left_v(grid, h, y):
    m, n = grid.shape
    i = int(y // h)
    if i < m:
        return grid[i,0]
    else:
        raise RuntimeError(f"i {i} is out of bounds for m {m}")

def right_v(grid, h, y):

    m, n = grid.shape
    i = int(y // h)
    if i < m:
        return grid[i, n-1]
    else:
        raise RuntimeError(f"i {i} is out of bounds for m {m}")


def sin_integral(a: float, b: float, denom: float, n: int):
    """
    Calculate integral of:

    np.sin((n * np.pi * x) / denom) from a to b.

    is:

        denom / (n * pi) * (cos((n pi a) / denom) - cos((n pi b) / denom))

    """

    return (denom / (n * np.pi)) * (np.cos((n * np.pi * a) / denom) - np.cos((n * np.pi * b) / denom))


def calculate_double_integral(func, x_low, x_high, y_low, y_high):


    area, error = dblquad(func, x_low, x_high, y_low, y_high, epsrel=1e-1, epsabs=1e-1)
    return area


class OriginalProblem(object):

    def __init__(self,
                 grid: np.ndarray,
                 h: float,
                 boundary=None):
        """
        The grid is provided below:

            [[2,3,2],
            [3,4,5],
            [4,5,6]]


        :param grid: np.array
        :param h: float, size of the grid
        """

        self.grid = grid
        m, n = self.grid.shape
        self.boundary = boundary
        self.h = h
        self.m = m
        self.n = n
        self.X = m * h
        self.Y = n * h


    def get_value(self, i, j):

        if 0 <= i and i < self.m and 0 <= j and j < self.n:
            return self.grid[i][j], (i*self.h, (i+1) * self.h), (j*self.h, (j+1)* self.h)
        else:
            raise ValueError("i and j are not in bounds")

    def get_boundary(self):
        """
        Returns boundary Function
        :return:
        """

        return self.boundary


    def add_mean_boundary(self, mean_boundary):
        self.boundary = mean_boundary


    def get_problem_size(self):

        return f"[0,{self.X}] x [0,{self.Y}]"


class LaplaceEquationSolver(object):

    def __init__(self, N, h: float, problem: LaplaceProblem):
        self.N = N
        self.h = h
        self.problem = problem

        self._setup_boundary(problem)

    def _check_boundary(self, u_sol):
        """
        Checks the boundary for the dataset.
        """

        # Compare u_sol to the boundary.

        max_x = self.problem.X
        max_y = self.problem.Y

        lower_boundary = self.lower_boundary
        upper_boundary = self.upper_boundary
        left_boundary = self.left_boundary
        right_boundary = self.right_boundary

        x_spacing = np.linspace(0, max_y, num=500, endpoint=False)
        y_spacing = np.linspace(0, max_x, num=500, endpoint=False)

        # Lower Horizontal Boundary
        plt.clf()
        plt.plot(x_spacing, [u_sol(x,0) for x in x_spacing])
        plt.plot(x_spacing, [lower_boundary(x) for x in x_spacing])
        plt.legend(["Calculated", "Actual"], loc="lower right")
        plt.title("Lower Boundary")
        plt.ylabel("Value")
        plt.xlabel("x")
        plt.savefig("lower_boundary_plot.png")

        # Upper Horizontal Boundary
        plt.clf()
        plt.plot(x_spacing, [u_sol(x, max_y) for x in x_spacing])
        plt.plot(x_spacing, [upper_boundary(x) for x in x_spacing])
        plt.legend(["Calculated", "Actual"], loc="lower right")
        plt.title("Upper Boundary")
        plt.ylabel("Value")
        plt.xlabel("x")
        plt.savefig("upper_boundary_plot.png")

        # Right Vertical Boundary
        plt.clf()
        plt.plot(y_spacing, [u_sol(max_x, y) for y in y_spacing])
        plt.plot(y_spacing, [right_boundary(y) for y in y_spacing])
        plt.legend(["Calculated", "Actual"], loc="lower right")
        plt.title("Right Boundary")
        plt.ylabel("Value")
        plt.xlabel("y")
        plt.savefig("right_boundary_plot.png")

        # Left Vertical Boundary
        plt.clf()
        plt.plot(y_spacing, [u_sol(0, y) for y in y_spacing])
        plt.plot(y_spacing, [left_boundary(y) for y in y_spacing])
        plt.legend(["Calculated", "Actual"], loc="lower right")
        plt.title("Left Boundary")
        plt.ylabel("Value")
        plt.xlabel("y")
        plt.savefig("left_boundary_plot.png")

        return True





    def calculate_fourier_coefficient_for_lower(self, f, n: int, use_analytical=USE_ANALYTICAL):

        def integrand(x):
            return f(x) * np.sin((n * np.pi * x) / self.problem.X)

        alpha = (2 / (self.problem.X * np.sinh((n * np.pi * self.problem.Y / self.problem.X))))
        if not use_analytical:
            numerical_value = alpha * quad(integrand, 0, self.problem.X)[0]
            return numerical_value
        else:
            res = 0
            for i in range(len(f.one_d_grid)):
                res += f.one_d_grid[i] * sin_integral(i * f.h, (i+1) * f.h, self.problem.X, n)
            analytical_value = alpha * res
            return analytical_value

    def _solve_lower_boundary(self, f) -> Callable[[float, float], float]:


        def u_1(x,y):
            res = 0
            for n in range(1, self.N + 1):
                a_n = self.calculate_fourier_coefficient_for_lower(f, n)
                #logging.debug(f"An is provided by {a_n}")
                val = np.sinh(n * np.pi * (self.problem.Y - y) / self.problem.X) * np.sin((n*np.pi*x)/self.problem.X)
                if not math.isfinite(val):
                    raise RuntimeError("Calculation is non finite")

                res += a_n * val

            return res

        return u_1



    def calculate_fourier_coefficient_for_upper(self, f: BoundaryManager, n: int, use_analytical=USE_ANALYTICAL):

        def integrand(x):

            return f(x) * np.sin((n * np.pi * x) / self.problem.X)

        alpha = (2 / (self.problem.X * np.sinh((n * np.pi * self.problem.Y / self.problem.X) )))
        if not use_analytical:
            return alpha * quad(integrand, 0, self.problem.X)[0]
        else:
            res = 0
            for i in range(len(f.one_d_grid)):
                res += f.one_d_grid[i] * sin_integral(i * f.h, (i + 1) * f.h, self.problem.X, n)
            analytical_value = alpha * res
            return analytical_value

    def _solve_upper_boundary(self, f) -> Callable[[float, float], float]:


        def u_2(x,y):
            res = 0
            for n in range(1, self.N + 1):
                b_n = self.calculate_fourier_coefficient_for_upper(f, n)
                #logging.debug(f"Bn is provided by {b_n}")
                val = np.sinh((n * np.pi * y) / self.problem.X) * np.sin((n * np.pi * x) / self.problem.X)

                if not math.isfinite(val):
                    raise RuntimeError("Calculation is non finite")

                res += b_n * val

            return res

        return u_2


    def calculate_fourier_coefficient_for_left(self, f, n: int, use_analytical=USE_ANALYTICAL):

        def integrand(y):
            return f(y) * np.sin((n * np.pi * y) / self.problem.Y)


        alpha = (2 / (self.problem.Y * np.sinh((n * np.pi * self.problem.X / self.problem.Y))))

        if not use_analytical:
            return alpha * quad(integrand, 0, self.problem.Y)[0]
        else:
            res = 0
            for i in range(len(f.one_d_grid)):
                res += f.one_d_grid[i] * sin_integral(i * f.h, (i + 1) * f.h, self.problem.Y, n)
            analytical_value = alpha * res
            return analytical_value


    def _solve_left_boundary(self, f) -> Callable[[float, float], float]:

        def u_3(x,y):
            res = 0
            for n in range(1, self.N + 1):
                c_n = self.calculate_fourier_coefficient_for_left(f, n)
                val = np.sinh((n * np.pi * (self.problem.X - x)) / self.problem.Y) * np.sin((n*np.pi*y)/self.problem.Y)

                if not math.isfinite(val):
                    raise RuntimeError("Calculation is non finite")

                #logging.debug(f"Cn is provided by {c_n}")
                res += c_n * val

            return res

        return u_3

    def calculate_fourier_coefficient_for_right(self, f, n: int, use_analytical=USE_ANALYTICAL):

        def integrand(y):
            return f(y) * np.sin((n * np.pi * y) / self.problem.Y)

        alpha = (2 / (self.problem.Y * np.sinh((n * np.pi * self.problem.X / self.problem.Y))))
        if not use_analytical:
            return alpha * quad(integrand, 0, self.problem.Y)[0]
        else:
            res = 0
            for i in range(len(f.one_d_grid)):
                res += f.one_d_grid[i] * sin_integral(i * f.h, (i + 1) * f.h, self.problem.Y, n)
            analytical_value = alpha * res
            return analytical_value

    def _solve_right_boundary(self, f) -> Callable[[float, float], float]:

        def u_4(x,y):
            res = 0
            for n in range(1, self.N + 1):
                d_n = self.calculate_fourier_coefficient_for_right(f, n)
                val = np.sinh((n*np.pi*x)/self.problem.Y) * np.sin((n*np.pi*y)/self.problem.Y)
                if not math.isfinite(val):
                    raise RuntimeError("Calculation is non finite")

                #logging.debug(f"Dn is provided by {d_n}")
                res += d_n * val

            return res

        return u_4

    def _setup_boundary(self, problem):

        upper_boundary = BoundaryManager(problem.boundary.high_h,
                                         DIRECTION.horizontal,
                                         EDGE.high,
                                         problem.grid,
                                         self.h)
        lower_boundary = BoundaryManager(problem.boundary.low_h,
                                         DIRECTION.horizontal,
                                         EDGE.low,
                                         problem.grid,
                                         self.h)

        left_boundary = BoundaryManager(problem.boundary.left_v,
                                        DIRECTION.vertical,
                                        EDGE.low,
                                        problem.grid,
                                        self.h)

        right_boundary = BoundaryManager(problem.boundary.right_v,
                                         DIRECTION.vertical,
                                         EDGE.high,
                                         problem.grid,
                                         self.h)

        self.upper_boundary = upper_boundary
        self.lower_boundary = lower_boundary
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        return upper_boundary, lower_boundary, left_boundary, right_boundary


    def solve(self, problem: LaplaceProblem) -> Callable[[float, float], float]:
        """
        Solves the laplace problem.

        The Laplace Equation for a 2D rectangular domain:

        u_xx + u_yy = 0


        :param problem:
        :return:
        """

        h = problem.h
        upper_boundary, lower_boundary, left_boundary, right_boundary = self._setup_boundary(problem)

        u_lower = self._solve_lower_boundary(lower_boundary)
        u_upper = self._solve_upper_boundary(upper_boundary)
        u_left = self._solve_left_boundary(left_boundary)
        u_right = self._solve_right_boundary(right_boundary)

        def result(x: float, y: float):
            return u_upper(x,y) + u_lower(x,y) + u_left(x,y) + u_right(x,y)


        return result




    def verification(self,
                     f: Callable[[float, float], float],
                     minimum,
                     maximum,
                     h):
        """
        Verification of the solved solution.

        An idea that I have for the verification is via plotting.

        """


        x_max = self.problem.X
        y_max = self.problem.Y

        x = np.linspace(0, x_max, 10)
        y = np.linspace(0, y_max, 10)

        X, Y = np.meshgrid(x, y, indexing='ij')

        ax = plt.figure().add_subplot(projection='3d')

        #X, Y = np.mgrid[0:6 * np.pi:0.25, 0:4 * np.pi:0.25]
        #Z = np.sqrt(np.abs(np.cos(X) + np.cos(Y)))


        Z = np.zeros(X.shape)
        m,n = Z.shape
        for i in range(m):
            for j in range(n):
                Z[i,j] = f(float(X[i,j]), float(Y[i,j]))

        verification_of_laplacian(f, self.problem.X, self.problem.Y, h, check_del = True)

        # Plot Boundaries
        boundary_correct = self._check_boundary(f)
        assert(boundary_correct)


        return Z


def calculate_modified_grid(original_problem, u_sol: Callable):

    # Use Laplace Solution to modify the grid.
    m, n = original_problem.grid.shape
    mod_grid = original_problem.grid.copy()
    for i in range(m):
        for j in range(n):
            logging.debug(f"Calculating Integral for Index {i} and index {j}".format(i=i, j=j))
            val, x_bounds, y_bounds = original_problem.get_value(i, j)
            x_low, x_high = x_bounds
            y_low, y_high = y_bounds
            integral = calculate_double_integral(u_sol, x_low, y_low, x_high, y_high)
            mod_grid[i][j] = mod_grid[i][j] - integral

    return mod_grid

class CalculusOfVariationsProblem(object):

    def __init__(self,
                 modified_grid: np.ndarray,
                 original_problem: OriginalProblem):

        self.h = original_problem.h
        self.X = original_problem.X
        self.Y = original_problem.Y
        self.modified_grid = modified_grid


def calculate_laplace_solution(grid = None,
                               h=None,
                               number_of_frequencies = 200):

    if grid is None and h is None:
        # Generate data
        grid_size = 100
        grid = generate_grid(grid_size)
        h = 0.1



    boundary = Boundary(low_h=partial(low_h, grid, h),
                        high_h=partial(high_h, grid, h),
                        left_v=partial(left_v, grid, h),
                        right_v=partial(right_v, grid, h))

    original_problem = OriginalProblem(grid, h, boundary=boundary)
    problem_size = original_problem.get_problem_size()

    laplace_problem = LaplaceProblem(original_problem.get_boundary(),
                                     h,
                                     original_problem.X,
                                     original_problem.Y,
                                     grid)

    solver = LaplaceEquationSolver(number_of_frequencies, h, laplace_problem)
    u_sol = solver.solve(laplace_problem)
    solver.verification(u_sol, grid.min(), grid.max(), h)

    return u_sol, original_problem


def calculate_laplace_and_grid(grid = None, h=None):

    u_sol, original_problem = calculate_laplace_solution(grid=grid, h=h)
    modified_grid = calculate_modified_grid(original_problem, u_sol)
    return u_sol, modified_grid


class CalculusOfVariationsSolver(object):

    def __init__(self):
        pass

    def solve(self, problem: CalculusOfVariationsProblem) -> Callable[[float], float]:
        """
        The solution to this problem will be based off of past code that I developed.

        The solution to this problem is similar to the previous code that was developed.


        :param problem:
        :return:
        """

        u, lambdas = create_two_dimensional_function(problem.h,
                                            problem.X,
                                            problem.Y,
                                            50,
                                            50,
                                            problem.modified_grid)

        return u











