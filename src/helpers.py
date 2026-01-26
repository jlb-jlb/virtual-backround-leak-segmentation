import numpy as np

import matplotlib.pyplot as plt


def show_image_from_tensor(img_tensor: np.ndarray):
    """
    Displays an image from a numpy tensor of shape (height, width, rgb).
    """
    if img_tensor.ndim != 3 or img_tensor.shape[2] != 3:
        raise ValueError("Input tensor must have shape (height, width, 3)")
    plt.imshow(img_tensor.astype(np.uint8))
    plt.axis("off")
    plt.show()
