"""
Load pixelated image and de-pixelate it.

Goal of the module is to understand the pixelated image, the size of the image, the number of pixels
and the number of required calculations.

The size will be required for the analysis of the required hardware.

"""
from pathlib import Path
import cv2
import os
import numpy as np
from scipy.integrate import dblquad
from calculus_of_variations_solver import create_two_dimensional_function
from variational_framework_2D import calculate_laplace_solution, calculate_modified_grid, calculate_laplace_and_grid
from verification import verification_of_laplacian


def divide_set(num_dim, i, total_set):


    if i >= num_dim:
        raise ValueError(f"i_x {i} is out of bounds for {num_dim}")

    if num_dim >= total_set:
        raise ValueError(f"num_dim {num_dim} is out of bounds for {total_set}")

    mod = total_set % num_dim
    divisor = total_set // num_dim
    if i < mod:
        low = (divisor + 1) * i
        high = (divisor + 1) * (i + 1)
        return low, high
    else:
        low = (divisor + 1) * mod + divisor * (i - mod)
        high = (divisor + 1) * mod + divisor * ((i + 1) - mod)
        return low, high



class Image(object):

    def __init__(self, image_name: str,
                 image_path: Path = None,
                 image= None):
        self.image_name = image_name
        self.image_path = image_path
        self.cv2_image = image
        if self.cv2_image is None:
            self._load_image()
        self.shape = self.cv2_image.shape
        self.x, self.y, self.channels = self.shape
        self.size = self.cv2_image.size

    def _load_image(self):
        if os.path.exists(self.image_path):
            self.cv2_image = cv2.imread(self.image_path)

    def get_cv2_image(self):
        return self.cv2_image

    def get_shape(self):
        return self.x, self.y, self.channels

    def get_sub_image(self,
                      num_x: int,
                      num_y: int,
                      i_x: int,
                      j_y: int):

        start_row, end_row = divide_set(num_x, i_x,self.x)
        start_col, end_col = divide_set(num_y, j_y,self.y)

        if self.cv2_image is not None:
            sub_image = self.cv2_image[start_row:end_row, start_col:end_col]
            return sub_image

    def __repr__(self):
        return (f"\n*********************************\n"
                f"Image Name: {self.image_name}\n"
                f"Number of Rows: {self.x}\n"
                f"Number of Columns: {self.y}\n"
                f"Number of Pixels: {self.x*self.y}\n"
                f"File Size: {self.size}")


def depixelation(img: Image):
    name = "pixelated_google_meet_image_of_colin"
    img = Image(name,
                "pixelated_image.png")
    engine = DepixelationEngine("depixelate", 10)
    sub_image = img.get_sub_image(100, 100, 3, 3)
    sub_image = Image("pixelated_sub_image", image=sub_image)
    new_img = engine.depixelate(sub_image)

def depixelation_with_boundary_correction(sub_image: np.ndarray):

    channel_0 = sub_image[:,:,0]
    channel_1 = sub_image[:,:,1]
    channel_2 = sub_image[:,:,2]


    u_sol_0, mod_grid_0 = calculate_laplace_and_grid(channel_0, 1)
    u_sol_1, mod_grid_1 = calculate_laplace_and_grid(channel_1, 1)
    u_sol_2, mod_grid_2 = calculate_laplace_and_grid(channel_2, 1)
    sub_image_0 = Image("pixelated_sub_image_with_laplace_correction", image=mod_grid_0)
    sub_image_1 = Image("pixelated_sub_image_with_laplace_correction", image=mod_grid_1)
    sub_image_2 = Image("pixelated_sub_image_with_laplace_correction", image=mod_grid_2)
    engine = DepixelationEngine("depixelate", 10)
    new_img_0 = engine.depixelate_without_laplacian(sub_image_0, func=u_sol_0)
    new_img_1 = engine.depixelate_without_laplacian(sub_image_1, func=u_sol_1)
    new_img_2 = engine.depixelate_without_laplacian(sub_image_2, func=u_sol_2)


class DepixelationEngine(object):

    def __init__(self, algo_name, grid_factor: int):
        self.algo_name = algo_name

    def depixelate_without_laplacian(self,
                                       image: Image,
                                       h=1.0,
                                       M=30,
                                       N=30,
                                       func=None):
        """
        Takes in an image or a subset of an image and aims to de-pixelate.

        :param image:
        :return:
        """

        num_channels = image.channels
        channel_to_grid = dict()
        new_grid_x = 200
        new_grid_y = 200
        x = np.linspace(0, image.x, new_grid_x)
        y = np.linspace(0, image.y, new_grid_y)
        h_x = image.x / new_grid_x
        h_y = image.y / new_grid_y
        new_image = np.zeros((new_grid_x, new_grid_y, num_channels), dtype=np.float32)
        for ch in range(num_channels):
            print("Processing channel ", ch)
            grid = image.get_cv2_image()[:,:,ch]

            u_sol,lambdas = create_two_dimensional_function(h,
                                                            image.x,
                                                            image.y,
                                                            M,
                                                            N,
                                                            grid)

            verification_of_laplacian(u_sol, image.x, image.y, h, lambdas, lambda x,y: 0, grid, M, N)

            for i, val_x in enumerate(x):
                for j, val_y in enumerate(y):
                    new_image[i,j,ch] = dblquad(u_sol, i*h_x, (i+1)*h_x, j*h_y, (j+1)*h_y) + dblquad(func, i*h_x, (i+1)*h_x, j*h_y, (j+1)*h_y)

        return new_image




if __name__ == "__main__":
    pass
