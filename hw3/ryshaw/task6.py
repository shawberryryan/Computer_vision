"""
Task6 Code
"""
import numpy as np
import common 
from common import save_img, read_img
from homography import fit_homography, homography_transform, RANSAC_fit_homography
import os
import cv2

def compute_distance(desc1, desc2):
    '''
    Calculates L2 distance between 2 binary descriptor vectors.
        
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
    
    Output - dist: a (N,M) L2 distance matrix where dist(i,j)
             is the squared Euclidean distance between row i of 
             desc1 and desc2. You may want to use the distance
             calculation trick
             ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    '''

    xNorm = np.sum(desc1**2, axis=-1, keepdims=True)
    yNorm = np.sum(desc2**2, axis=-1, keepdims=True)
    distances = xNorm + yNorm.T - 2 * np.dot(desc1, desc2.T)
    return np.sqrt(np.maximum(distances, 0))

def find_matches(desc1, desc2, ratioThreshold):
    '''
    Calculates the matches between the two sets of keypoint
    descriptors based on distance and ratio test. Using function compute_distance(desc1, desc2)
    
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
            ratioThreshhold : maximum acceptable distance ratio between 2
                              nearest matches 
    
    Output - matches: a list of indices (i,j) 1 <= i <= N, 1 <= j <= M giving
             the matches between desc1 and desc2.
             
             This should be of size (K,2) where K is the number of 
             matches and the row [ii,jj] should appear if desc1[ii,:] and 
             desc2[jj,:] match.
    '''
    # Compute distance matrix
    dist_matrix = compute_distance(desc1, desc2)

    # Find best and second best matches for each descriptor in desc1
    best_matches = np.argmin(dist_matrix, axis=1)
    best_dist = dist_matrix[np.arange(len(desc1)), best_matches]
    second_best_matches = np.argsort(dist_matrix, axis=1)[:, 1]
    second_best_dist = dist_matrix[np.arange(len(desc1)), second_best_matches]

    # Apply ratio test to find valid matches
    valid_matches = np.where(best_dist < ratioThreshold * second_best_dist)[0]

    # Construct list of indices of valid matches
    matches = np.stack((valid_matches, best_matches[valid_matches]), axis=1)

    return matches

def draw_matches(img1, img2, kp1, kp2, matches):
    '''
    Creates an output image where the two source images stacked vertically
    connecting matching keypoints with a line. 
        
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            kp1: Keypoint matrix for image 1 of shape (N,4)
            kp2: Keypoint matrix for image 2 of shape (M,4)
            matches: List of matching pairs indices between the 2 sets of 
                     keypoints (K,2)
    
    Output - Image where 2 input images stacked vertically with lines joining 
             the matched keypoints
    Hint: see cv2.line
    '''
    # Hint:
    # Use common.get_match_points() to extract keypoint locations
    # Extract keypoint locations
    match_points = common.get_match_points(kp1, kp2, matches)
    p1, p2 = match_points[:, :2], match_points[:, 2:]

    # Compute the shape of the output image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    output_height = h1 + h2
    output_width = max(w1, w2)

    # Create the output image
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Copy the input images to the output image
    output_image[:h1, :w1, :] = img1
    output_image[h1:, :w2, :] = img2

    # Adjust the location of keypoints in the second image
    p2[:, 1] += h1

    # Draw lines between the matched keypoints
    for i in range(len(matches)):
        point1 = (int(p1[i][0]), int(p1[i][1]))
        point2 = (int(p2[i][0]), int(p2[i][1]))
        cv2.line(output_image, point1, point2, (0, 255, 0), 1)

    return output_image


def warp_and_combine(img1, img2, H):
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
    (b) Pick which image you’re going to merge to; without loss of generality, pick image 1. 
    Figure out how to make a merged image that’s big enough to hold both image 1 and transformed image 2. 
    Think of this as finding the smallest enclosing rectangle of both images. 
    The upper left corner of this rectangle (i.e., pixel [0, 0]) may not be at the same location as in image 1. 
    You will almost certainly need to hand-make a homography that translates image 1 to its location in the merged image. 
    For doing this calculations, use the fact that the image content will be bounded by the image corners. 
    Looking at the min, max of these gives you what you need to create the panorama. 
    (c) Warp both images to the merged image. 
    You can figure out where the images go by warping images containing ones to the merged images instead of the image and 
    filling the image with 0s where the image doesn’t go. These masks also tell you how to create the average.
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
    output_image[overlap] = output_image[overlap] / 2
    return output_image
     



def make_warped(img1, img2):
    '''
    Take two images and return an image, putting together the full pipeline.
    You should return an image of the panorama put together.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 1 of shape (H2,W2,3)
    
    Output - Final stitched image
    Be careful about:
    a) The final image size 
    b) Writing code so that you first estimate H and then merge images with H.
    The system can fail to work due to either failing to find the homography or
    failing to merge things correctly.
    '''
    k1, d1 = common.get_AKAZE(img1)
    k2, d2 = common.get_AKAZE(img2)
    matches = find_matches(d1, d2, 0.8)
    matchedPoints = common.get_match_points(k1, k2, matches)
    H = RANSAC_fit_homography(matchedPoints)
    stitched = warp_and_combine(img1, img2, H)
    return stitched 


if __name__ == "__main__":

    #Possible starter code; you might want to loop over the task 6 images
    to_stitch = 'florence2'
    I1 = read_img(os.path.join('task6',to_stitch,'p1.jpg'))
    I2 = read_img(os.path.join('task6',to_stitch,'p2.jpg'))
    res = make_warped(I1,I2)
    save_img(res,"result_"+to_stitch+".jpg")
