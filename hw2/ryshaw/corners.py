import os

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    offset_image = np.roll(np.roll(image, u, axis=0), v, axis=1)
    diff_image = (image - offset_image)**2
    window = np.ones(window_size)
    return scipy.ndimage.convolve(diff_image, window, mode='constant')

def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    Ix = np.zeros(image.shape)
    Iy = np.zeros(image.shape)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ix = scipy.ndimage.convolve(image, kernel)
    Iy = scipy.ndimage.convolve(image, kernel.T)

    Ixx = scipy.ndimage.gaussian_filter(Ix * Ix, sigma=0.833)
    Iyy = scipy.ndimage.gaussian_filter(Iy * Iy, sigma=0.833)
    Ixy = scipy.ndimage.gaussian_filter(Ix * Iy, sigma=0.833)

    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    alpha = 0.05

    return det - alpha * trace**2


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 6: Corner Score --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calulcate corner score
    u, v, W = 0, 5, (5,5)

    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score.png")

    # Computing the corner scores for various u, v values.
    score = corner_score(img, 0, 5, W)
    save_img(score, "./feature_detection/corner_score05.png")

    score = corner_score(img, 0, -5, W)
    save_img(score, "./feature_detection/corner_score0-5.png")

    score = corner_score(img, 5, 0, W)
    save_img(score, "./feature_detection/corner_score50.png")

    score = corner_score(img, -5, 0, W)
    save_img(score, "./feature_detection/corner_score-50.png")

    # (c): No Code

    # -- TODO Task 7: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")

    fig, ax = plt.subplots()
    image = ax.imshow(harris_corners, cmap='magma')
    fig.colorbar(image)
    plt.savefig("./feature_detection/res.png")


if __name__ == "__main__":
    main()
