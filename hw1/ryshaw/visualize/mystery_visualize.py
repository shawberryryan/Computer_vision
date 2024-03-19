#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    H, W = X.shape
    # Normalize the values of X between 0 and 1
    X = (X - X.min()) / (X.max() - X.min())
    # Rescale the values of X to the range of 0 to N-1
    X = (X * (colors.shape[0] - 1)).astype(np.uint8)
    # Use the colormap to map the values of X to RGB values
    image = colors[X]
    # Convert the image to uint8
    image = (image * 255).astype(np.uint8)
    return image


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata.npy")
    for i in range(9):
        plt.imsave("vis_%d.png" % i, colormapArray(data[:,:,i],colors))

    pdb.set_trace()
