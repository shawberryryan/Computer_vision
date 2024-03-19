"""
Homography fitting functions
You should write these
"""
import numpy as np
from common import homography_transform

def fit_homography(XY):
        '''
        Given a set of N correspondences XY of the form [x,y,x',y'],
        fit a homography from [x,y,1] to [x',y',1].

        Input - XY: an array with size(N,4), each row contains two
                points in the form [x_i, y_i, x'_i, y'_i] (1,4)
        Output -H: a (3,3) homography matrix that (if the correspondences can be
                described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T
        '''
        #create matrix A where we have p_i = [x_i, y_i, 1] and 0.T = [0, 0, 0]
        #arrange the rows of A in the form row 2i: [0.T, -(p_i), y'_i*p_i] row 2i+1: [p_i, 0.T, -x'_i*p_i]
        A = np.zeros((XY.shape[0]*2,9))
        for i in range(XY.shape[0]):
                p_i = np.array([XY[i,0],XY[i,1],1])
                A[2*i] = np.array([0,0,0,-p_i[0],-p_i[1],-p_i[2],XY[i,3]*p_i[0],XY[i,3]*p_i[1],XY[i,3]*p_i[2]])
                A[2*i+1] = np.array([p_i[0],p_i[1],p_i[2],0,0,0,-XY[i,2]*p_i[0],-XY[i,2]*p_i[1],-XY[i,2]*p_i[2]])


        #solve for h in Ah=0
        _, _, V = np.linalg.svd(A)
        h = V[-1]
        H = np.reshape(h,(3,3))
        #normalize last entry to 1
        H = H/H[2,2]
        return H


def RANSAC_fit_homography(XY, eps=1, nIters=1000):
    '''
    Perform RANSAC to find the homography transformation 
    matrix which has the most inliers
        
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            eps: threshold distance for inlier calculation
            nIters: number of iteration for running RANSAC
    Output - bestH: a (3,3) homography matrix fit to the 
                    inliers from the best model.

    Hints:
    a) Sample without replacement. Otherwise you risk picking a set of points
       that have a duplicate.
    b) *Re-fit* the homography after you have found the best inliers
    '''
    bestH, bestCount, bestInliers = np.eye(3), -1, np.zeros((XY.shape[0],))
    for i in range(nIters):
        subset = np.random.choice(XY.shape[0],4,replace=False)
        model = fit_homography(XY[subset])
        error = homography_transform(XY[:,:2],model) - XY[:,2:]
        inliers = np.linalg.norm(error,axis=1) < eps
        if np.sum(inliers) > bestCount:
                bestCount = np.sum(inliers)
                bestInliers = inliers
                bestH = fit_homography(XY[bestInliers])
    # apply bestH to inliers to get bestRefit
    bestRefit = fit_homography(XY[bestInliers])
    return bestRefit

if __name__ == "__main__":
    #If you want to test your homography, you may want write any code here, safely
    #enclosed by a if __name__ == "__main__": . This will ensure that if you import
    #the code, you don't run your test code too
    pass
