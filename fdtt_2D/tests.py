"""
Module tests the solver that are provided.

"""


import unittest
from variational_framework_2D import (generate_grid,
                                      Boundary,
                                      OriginalProblem,
                                      LaplaceProblem,
                                      LaplaceEquationSolver,
                                      left_v,
                                      right_v,
                                      low_h,
                                      high_h,
                                      BoundaryManager,
                                      generate_zero_grid,
                                      CalculusOfVariationsProblem,
                                      CalculusOfVariationsSolver,
                                      calculate_modified_grid,
                                      )

from image import depixelation_with_boundary_correction
from typing import Callable, List
import numpy as np
import math
from collections.abc import Callable
from typing import Protocol
from functools import partial
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import logging
from calculus_of_variations_solver import integral_sin
from image import divide_set
from image import Image, DepixelationEngine



class TestLaplaceSolver(unittest.TestCase):

    def test_zero_boundary_laplace_problem(self):
        # Generate data
        grid_size = 10
        grid = generate_zero_grid(grid_size)
        h = 0.1

        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

        def func(x, grid, h):
            return 0

        boundary = Boundary(low_h=partial(func, grid, h),
                            high_h=partial(func, grid, h),
                            left_v=partial(func, grid, h),
                            right_v=partial(func, grid, h))

        original_problem = OriginalProblem(grid, h, boundary=boundary)
        problem_size = original_problem.get_problem_size()
        laplace_problem = LaplaceProblem(original_problem.get_boundary(),
                                         h,
                                         original_problem.X,
                                         original_problem.Y,
                                         grid)
        number_of_frequencies = 20
        solver = LaplaceEquationSolver(number_of_frequencies, h, laplace_problem)
        u_sol = solver.solve(laplace_problem)
        Z = solver.verification(u_sol, grid.min(), grid.max(),h)

        self.assertTrue(np.isclose(Z.max(), 0.0))


    def test_non_zero_laplace_problem(self):
        # Generate data
        grid_size = 50
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

        #TODO: Change this number to get better results.
        number_of_frequencies = 200
        solver = LaplaceEquationSolver(number_of_frequencies, h, laplace_problem)
        u_sol = solver.solve(laplace_problem)
        solver.verification(u_sol, grid.min(), grid.max(), h)
        modified_grid = calculate_modified_grid(original_problem, u_sol) #TODO: Modify
        calculus_of_variations_problem = CalculusOfVariationsProblem(modified_grid, original_problem)
        calculus_of_variations_solver = CalculusOfVariationsSolver()
        calculus_of_variations_solver.solve(calculus_of_variations_problem)

    def test_non_zero_laplace_problem_non_square_grid(self):

        #Number of rows and columns are similar but not equal
        #there is no overflow so things still work.

        # Generate data
        grid_size = 50
        grid = generate_grid(grid_size, x=grid_size, y=grid_size+1)
        h = 1

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

        #TODO: Change this number to get better results.
        number_of_frequencies = 200
        solver = LaplaceEquationSolver(number_of_frequencies, h, laplace_problem)
        u_sol = solver.solve(laplace_problem)
        solver.verification(u_sol, grid.min(), grid.max(), h)
        modified_grid = calculate_modified_grid(original_problem, u_sol) #TODO: Modify
        calculus_of_variations_problem = CalculusOfVariationsProblem(modified_grid, original_problem)
        calculus_of_variations_solver = CalculusOfVariationsSolver()
        calculus_of_variations_solver.solve(calculus_of_variations_problem)

    def test_non_zero_laplace_problem_non_square_grid(self):

        #when the number of rows is not similar to the number of
        #columns. in this example, we demonstrate this.
        #there is an issue with overflow in sinh.
        #this causes everything to break down.

        # Generate data
        grid_size = 50
        grid = generate_grid(grid_size, x=grid_size, y=2*grid_size)
        h = 1

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

        #TODO: Change this number to get better results.
        number_of_frequencies = 200
        solver = LaplaceEquationSolver(number_of_frequencies, h, laplace_problem)
        u_sol = solver.solve(laplace_problem)
        solver.verification(u_sol, grid.min(), grid.max(), h)
        modified_grid = calculate_modified_grid(original_problem, u_sol) #TODO: Modify
        calculus_of_variations_problem = CalculusOfVariationsProblem(modified_grid, original_problem)
        calculus_of_variations_solver = CalculusOfVariationsSolver()
        calculus_of_variations_solver.solve(calculus_of_variations_problem)


    def test_calculate_fourier_coefficient(self):

        # Generate data
        grid_size = 1000
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
        number_of_frequencies = 500
        solver = LaplaceEquationSolver(number_of_frequencies, h, laplace_problem)
        upper_boundary, lower_boundary, left_boundary, right_boundary = solver._setup_boundary(laplace_problem)
        analytical_coefficient = solver.calculate_fourier_coefficient_for_lower(lower_boundary,
                                                                        4,
                                                                           use_analytical=True)

        numerical_coefficient = solver.calculate_fourier_coefficient_for_lower(lower_boundary,
                                                                          4,
                                                                          use_analytical=False)

        self.assertTrue(np.isclose(analytical_coefficient, numerical_coefficient, rtol=0.03, atol=0.01))

        analytical_coefficient = solver.calculate_fourier_coefficient_for_upper(upper_boundary,
                                                                                4,
                                                                                use_analytical=True)

        numerical_coefficient = solver.calculate_fourier_coefficient_for_upper(upper_boundary,
                                                                               4,
                                                                               use_analytical=False)



        self.assertTrue(np.isclose(analytical_coefficient, numerical_coefficient, rtol=0.03, atol=0.01))

        analytical_coefficient = solver.calculate_fourier_coefficient_for_left(upper_boundary,
                                                                                4,
                                                                                use_analytical=True)

        numerical_coefficient = solver.calculate_fourier_coefficient_for_left(upper_boundary,
                                                                               4,
                                                                               use_analytical=False)

        self.assertTrue(np.isclose(analytical_coefficient, numerical_coefficient, rtol=0.03, atol=0.01))

        analytical_coefficient = solver.calculate_fourier_coefficient_for_right(upper_boundary,
                                                                               4,
                                                                               use_analytical=True)

        numerical_coefficient = solver.calculate_fourier_coefficient_for_right(upper_boundary,
                                                                              4,
                                                                              use_analytical=False)

        self.assertTrue(np.isclose(analytical_coefficient, numerical_coefficient, rtol=0.03, atol=0.01))


    def test_create_two_dimensional_function_with_zero_data(self):

        from calculus_of_variations_solver import create_two_dimensional_function
        from verification import verification_of_laplacian
        L_x = 5
        L_y = 5
        M = 300
        N = 300
        h = 1
        grid = np.zeros((int(L_x //h), int(L_y //h)), dtype=np.float64)

        #Save the result, so we can fix the results for future.

        u, lambdas = create_two_dimensional_function(h,
                                                    L_x,
                                                    L_y,
                                                    M,
                                                    N,
                                                    grid)

        func = lambda x,y: 0.0
        # Verification of the two d function.
        verification_of_laplacian(u, L_x, L_y,h,lambdas, func, grid, M, N)


    def test_create_two_dimensional_function_with_identity_data(self):

        from calculus_of_variations_solver import create_two_dimensional_function
        from verification import verification_of_laplacian
        L_x = 5
        L_y = 5
        M = 300
        N = 300
        h = 1
        grid = np.eye(int(L_x //h), dtype=np.float64)

        #Save the result, so we can fix the results for future.

        u, lambdas = create_two_dimensional_function(h,
                                                    L_x,
                                                    L_y,
                                                    M,
                                                    N,
                                                    grid)

        func = lambda x,y: 0.0
        # Verification of the two d function.
        verification_of_laplacian(u, L_x, L_y,h,lambdas, func, grid, M, N)


    def test_create_two_dimensional_function_with_random_data(self):

        from calculus_of_variations_solver import create_two_dimensional_function
        from verification import verification_of_laplacian
        L_x = 5
        L_y = 5
        M = 300
        N = 300
        h = 1
        grid = np.random.rand(int(L_x //h), int(L_y //h))

        #Save the result, so we can fix the results for future.

        u, lambdas = create_two_dimensional_function(h,
                                                    L_x,
                                                    L_y,
                                                    M,
                                                    N,
                                                    grid)

        func = lambda x,y: 0.0
        # Verification of the two d function.
        verification_of_laplacian(u, L_x, L_y,h,lambdas, func, grid, M, N)

    def test_create_two_dimensional_function_with_random_data_and_non_square_grid(self):
        #TODO: This fails because there is a memory issue

        from calculus_of_variations_solver import create_two_dimensional_function
        from verification import verification_of_laplacian
        L_x = 947
        L_y = 1792
        M = 10
        N = 10
        h = 1
        grid = np.random.rand(int(L_x // h), int(L_y // h))

        # Save the result, so we can fix the results for future.

        u, lambdas = create_two_dimensional_function(h,
                                                     L_x,
                                                     L_y,
                                                     M,
                                                     N,
                                                     grid)

        func = lambda x, y: 0.0
        # Verification of the two d function.
        verification_of_laplacian(u, L_x, L_y, h, lambdas, func, grid, M, N)



    def test_create_two_dimensional_function_with_random_data_and_non_square_grid_one_tenth(self):
        """
        The full pixelated image requires:
            L_x = 947
            L_y = 1792

        I divide this into 10 rows and 10 columns. And solve each one individually.

        :return:
        """

        from calculus_of_variations_solver import create_two_dimensional_function
        from verification import verification_of_laplacian
        L_x = 947 // 100
        L_y = 1792 // 100
        M = 30
        N = 30
        h = 1
        grid = np.random.rand(int(L_x // h), int(L_y // h))

        # Save the result, so we can fix the results for future.

        u, lambdas = create_two_dimensional_function(h,
                                                     L_x,
                                                     L_y,
                                                     M,
                                                     N,
                                                     grid)

        func = lambda x, y: 0.0
        # Verification of the two d function.
        verification_of_laplacian(u, L_x, L_y, h, lambdas, func, grid, M, N)



    def test_sin_integral(self):



        x = 4
        h = 0.1
        m = 2
        L = 5
        analytical_integral = integral_sin(x, h, m, L)
        actual_integral, error = quad(lambda x: math.sin(m * x * np.pi/L), x*h, (x+1)*h)


        self.assertTrue(np.isclose(analytical_integral, actual_integral))


    def test_divide_set(self):

        num_dim = 10
        i = 7
        total_set = 998
        low, high = divide_set(num_dim, i, total_set)

    def test_depixelation(self):

        name = "pixelated_google_meet_image_of_colin"
        img = Image(name,
                    "pixelated_image.png")
        engine = DepixelationEngine("depixelate", 10)
        sub_image = img.get_sub_image(100, 100, 3, 3)
        sub_image = Image("pixelated_sub_image", image=sub_image)
        new_img = engine.depixelate(sub_image)
        self.assertTrue(np.all(new_img > 0), "Pixelated image does not solely contain non-negative values.")


    def test_depixelation_with_laplace_correction(self):
        depixelation_with_boundary_correction()





if __name__ == '__main__':
    unittest.main()