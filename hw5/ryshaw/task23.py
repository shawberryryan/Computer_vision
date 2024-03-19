from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import pdb
import os


def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """

    #This will give you an answer you can compare with
    #Your answer should match closely once you've divided by the last entry
    FOpenCV, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    # Step 1: Compute scaling matrix T = [[1/s,0, -1/2], [0, 1/s, -1/2], [0,0,1]] where s = sqrt(h^2 + w^2)
    s = np.sqrt(shape[0]**2 + shape[1]**2)
    T = np.array([[1/s, 0, -1/2], [0, 1/s, -1/2], [0, 0, 1]])

    # Step 2: compute scaled points P = {(ui,vi)} and P' = {(ui',vi')} by applying T to pts1 and pts2
    P = T@homogenize(pts1).T
    P_prime = T@homogenize(pts2).T

    # Step 3: Compute U where each row is [x'i*xi, x'i*yi, x'i, y'i*xi, y'i*yi, y'i, xi, yi, 1], using P and P'; solve for F_init via eigenvectors of UU^T. 
    # F_init is "wrong" for two reasons: it might be full rank, and it "works" for the scaled points, not the original points.
    U = np.zeros((P.shape[1], 9))
    for i in range(P.shape[1]):
        U[i] = [P_prime[0,i]*P[0,i], P_prime[0,i]*P[1,i], P_prime[0,i], P_prime[1,i]*P[0,i], P_prime[1,i]*P[1,i], P_prime[1,i], P[0,i], P[1,i], 1]
    UU = U.T@U
    _, V = np.linalg.eig(UU)
    F_init = V[:, -1].reshape(3,3)

    # Step 4: Rank-redule F_init using SVD: construct the decomposition F_init = USV^T; set S_ to S, but with the last entry set to 0; then compute F_rank2 = US_V^T.
    # F_rank2 by construction has rank 2, fixing the first problem with F_init.
    U, S, V = np.linalg.svd(F_init)
    S_ = np.diag(S)
    S_[-1, -1] = 0
    F_rank2 = U@S_@V

    # Step 5: Return F = T^T F_rank2 T. Since F gets used as [u',v',1]^T F [u,v,1], the matrix can be thought of as applying T to either point before it's used by the
    # fundamental matrix in the middle. The transpose on the left is critical: [u',v',1]^T T^T = (T[u',v',1])^T and so the transpose is needed to get the right answer.
    F = T.T@F_rank2@T

    # Step 6: normalize the data by scale the image size and centering the data at 0
    F = F/F[2,2]

    # print("F OpenCV: ", FOpenCV)
    # print("F: ", F)

    return F

def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    #return the homogenous coordinates of the epipoles
    U, S, V = np.linalg.svd(F)
    e1 = V[-1]
    e2 = U[:,-1]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1, e2


def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices. Let X be a
    point in 3D in homogeneous coordinates. For two cameras, we have

        p1 === M1 X
        p2 === M2 X

    Triangulation is to solve for X given p1, p2, M1, M2.

    Inputs:
    - K1: Numpy array of shape (3,3) giving camera instrinsic matrix for img1
    - K2: Numpy array of shape (3,3) giving camera instrinsic matrix for img2
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - pcd: Numpy array of shape (N,4) giving the homogeneous 3D point cloud
      data
    """
    pcd = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    #cast K1 and K2 to float64
    K1 = K1.astype(np.float64)
    K2 = K2.astype(np.float64)
    

    # Get the essential matrix
    E = K2.T @ F @ K1

    # Decompose the essential matrix to get R and t
    R1, R2, t = cv2.decomposeEssentialMat(E)

    #The first camera's projectiom matrix is M1 = (K1@I, K1@0) where I is the identity matrix
    M1 = np.hstack((K1@np.identity(3), K1@np.zeros(t.shape)))

    # Possible camera matrices for M2
    M2_options = [K2@np.hstack((R1, t)),
                  K2@np.hstack((R1, -t)),
                  K2@np.hstack((R2, t)),
                  K2@np.hstack((R2, -t))]

    # try all the M2 and figure out which one puts the most points in front of the camera;
    # then do the triangulation to get the final set of points
    max_count = 0
    for M2 in M2_options:
        # Triangulate the points
        points4D = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)
        points3D = (points4D / points4D[-1]).T


        # Count the number of points in front of the camera
        count = np.count_nonzero(points3D[:,2] > 0)

        # Update the max_count and pcd
        if count > max_count:
            max_count = count
            pcd = -points3D
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pcd

if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape

        # compute F
        F = find_fundamental_matrix(shape, pts1, pts2)
        FOpenCV, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
        # compute the epipoles
        e1, e2 = compute_epipoles(F)
        print(e1, e2)
        #to get the real coordinates, divide by the last entry
        print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        outname = os.path.join(output, name + "_us.png")
        # If filename isn't provided or is None, this plt.shows().
        # If it's provided, it saves it
        draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
                      epi1=e1, epi2=e2, filename=outname)
        
        # # Using F and K1 and K2 print the essential matrix
        # print("Essential Matrix: ", K2.T@F@K1)

        if 1:
            #you can turn this on or off
            pcd = find_triangulation(K1, K2, F, pts1, pts2)
            visualize_pcd(pcd.T,
                          filename=os.path.join(output, name + "_rec.png"))


