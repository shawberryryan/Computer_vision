"""
Task 7 Code
"""
import numpy as np
import common 
from common import save_img, read_img
from homography import homography_transform, RANSAC_fit_homography
from task6 import find_matches, warp_and_combine
import cv2
import os

def task7_warp_and_combine(img1, img2, H):
    '''
    You may want to write a function that merges the two images together given
    the two images and a homography: once you have the homography you do not
    need the correspondences; you just need the homography.
    Writing a function like this is entirely optional, but may reduce the chance
    of having a bug where your homography estimation and warping code have odd
    interactions.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            H: homography mapping betwen them
    Output - V: stitched image of size (?,?,3); unknown since it depends on H
                but make sure in V, for pixels covered by both img1 and warped img2,
                you see only img2
    '''
    H = np.linalg.inv(H)
    # Get the height and width of each input image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.array([[0, 0, 1],
                        [w2, 0, 1],
                        [w2, h2, 1],
                        [0, h2, 1]])
    
    corners = np.dot(H, corners.T).T
    corners = (corners / corners[:, 2:]).astype(int)

    x_min = int(np.minimum(np.min(corners[:, 0]), 0))
    x_max = int(np.maximum(np.max(corners[:, 0]), w1))
    y_min = int(np.minimum(np.min(corners[:, 1]), 0))
    y_max = int(np.maximum(np.max(corners[:, 1]), h1))

    # Create the output image
    output_image = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint64)

    # Translation
    tx, ty = np.absolute(x_min), np.absolute(y_min)
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]]).astype(np.float64)
    H = np.dot(T, H)
    H = H / H[2, 2]

    # Warp the first image
    img1 = cv2.warpPerspective(img1, T, (output_image.shape[1], output_image.shape[0]))
    img2 = cv2.warpPerspective(img2, H, (output_image.shape[1], output_image.shape[0]))

    # Create the mask
    mask1, mask2 = img1 != 0, img2 != 0
    overlap = np.logical_and(mask1, mask2)
    output_image = output_image + img1 + img2
    output_image[overlap] = output_image[overlap] - img1[overlap]
    return output_image

def improve_image(scene, template, transfer):
    '''
    Detect template image in the scene image and replace it with transfer image.

    Input - scene: image (H,W,3)
            template: image (K,K,3)
            transfer: image (L,L,3)
    Output - augment: the image with 
    
    Hints:
    a) You may assume that the template and transfer are both squares.
    b) This will work better if you find a nearest neighbor for every template
       keypoint as opposed to the opposite, but be careful about directions of the
       estimated homography and warping!
    '''
    #scale transder image to template size
    transfer = cv2.resize(transfer, (template.shape[0], template.shape[1]))

    k1, d1 = common.get_AKAZE(scene)
    k2, d2 = common.get_AKAZE(template)
    matches = find_matches(d1, d2, 0.7)
    matched_points = common.get_match_points(k1, k2, matches)
    H = RANSAC_fit_homography(matched_points)
    augment = task7_warp_and_combine(scene, transfer, H)
    return augment

if __name__ == "__main__":
    # Task 7


    scene = read_img(os.path.join('myscene.jpg'))
    template = read_img(os.path.join('mytemplate.jpg'))

    transfer = read_img(os.path.join('mytransfer.jpg'))
    augment = improve_image(scene, template, transfer)

    save_img(augment,"myimproved.jpg")

