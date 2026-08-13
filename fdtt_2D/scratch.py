import numpy as np


def example_1():
    # 1. Create 1D arrays for the x and y axes with evenly spaced values
    x = np.linspace(-5, 5, 5)  # 5 points between -5 and 5
    y = np.linspace(-2, 2, 4)  # 4 points between -2 and 2

    # 2. Create the 2D grid matrices
    xv, yv = np.meshgrid(x, y)

    print(xv, yv)

    import matplotlib.pyplot as plt
    plt.plot(xv, yv, marker='o', color='k', linestyle='none')
    plt.show()


def example_2():
    import numpy as np
    from PIL import Image

    # 1. Load your image file
    image_path = "cat.jpg"  # Replace with your file
    img = Image.open(image_path)

    # 2. Convert the image to a NumPy array (Grid format)
    # For RGB images, this creates a 3D grid: [Rows, Columns, Color Channels]
    imageGrid = np.array(img)

    # 3. View the grid shape and a small chunk of data
    print(f"Grid Shape (Height, Width, Channels): {imageGrid.shape}")
    print("Top-left pixel grid block data (RGB values):")
    print(imageGrid[0:3, 0:3])  # Displays a 3x3 pixel grid slice

    print(imageGrid.shape)


def example_3():

    import matplotlib.pyplot as plt
    import numpy as np

    # Generate random points
    x = np.random.rand(50)
    y = np.random.rand(50)
    colors = np.random.rand(50)

    # Create scatter plot with variable colors
    plt.scatter(x, y, c=colors, cmap='viridis', s=100, alpha=0.8)
    plt.title('2D Scatter Plot')
    plt.colorbar(label='Intensity')  # Adds a scale bar
    plt.show()



def example_4():
    import matplotlib.pyplot as plt
    import numpy as np

    ax = plt.figure().add_subplot(projection='3d')

    X, Y = np.mgrid[0:6 * np.pi:0.25, 0:4 * np.pi:0.25]
    Z = np.sqrt(np.abs(np.cos(X) + np.cos(Y)))

    ax.plot_surface(X + 1e5, Y + 1e5, Z, cmap='autumn', cstride=2, rstride=2)

    ax.set_xlabel("X label")
    ax.set_ylabel("Y label")
    ax.set_zlabel("Z label")
    ax.set_zlim(0, 2)

    plt.show()


if __name__ == '__main__':
    example_4()