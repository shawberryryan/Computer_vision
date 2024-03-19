import numpy as np
import utils


def find_projection(pts2d, pts3d):
    """
    Computes camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    [u v 1]^T === M [x y z 1]^T

    Where (u,v) are the 2D image coordinates and (x,y,z) are the world 3D
    coordinates

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - M: Numpy array of shape (3,4)

    """
    M = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    assert pts2d.shape[0] == pts3d.shape[0], "Number of 2D and 3D points must match"

    # Convert points to homogeneous coordinates
    pts2d_hom = utils.homogenize(pts2d)
    pts3d_hom = utils.homogenize(pts3d)

    # Solve the linear system Ax=b using the SVD method
    A = np.zeros((2 * pts2d.shape[0], 12))
    for i in range(pts2d.shape[0]):
        A[2 * i, :4] = pts3d_hom[i]
        A[2 * i, 8:12] = pts3d_hom[i]*-pts2d_hom[i, 0]
        A[2 * i + 1, 4:8] = pts3d_hom[i]
        A[2 * i + 1, 8:12] = pts3d_hom[i]*-pts2d_hom[i, 1]
    U, S, Vt = np.linalg.svd(A)
    M = Vt[-1, :].reshape(3, 4)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return M

def compute_distance(pts2d, pts3d):
    """
    use find_projection to find matrix M, then use M to compute the average 
    distance in the image plane (i.e., pixel locations) 
    between the homogeneous points M X_i and 2D image coordinates p_i

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - float: a average distance you calculated (threshold is 0.01)

    """
    distance = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    M = find_projection(pts2d, pts3d)
    pts3d_hom = utils.homogenize(pts3d)
    pts2d_hom_pred = np.dot(M, pts3d_hom.T).T
    pts2d_pred = utils.dehomogenize(pts2d_hom_pred)
    distance = np.mean(np.linalg.norm(pts2d_pred - pts2d, axis=1))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distance

if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    """
    data = np.load("task23/ztrans/data.npz")
    pts2d = data['pts1']
    pts3d = data['pts1_3D']
    """
    print(find_projection(pts2d, pts3d))
    
    foundDistance = compute_distance(pts2d, pts3d)
    print("Distance: %f" % foundDistance)
