import numpy as np
from matplotlib import pyplot as plt
from homography import fit_homography, homography_transform, RANSAC_fit_homography

def p3():
    # load points X from task3/ [x1,y1,x1',y1']
    X = np.load('task3/points_case_2.npy')
    x = X[:,:2]

    # create matrix A of R^2kx6 which is of the format [x_i,y_i,0,0,1,0],[0,0,x_i,y_i,0,1]
    A = np.zeros((X.shape[0]*2,6))
    for i in range(X.shape[0]):
        A[2*i] = [X[i,0],X[i,1],0,0,1,0]
        A[2*i+1] = [0,0,X[i,0],X[i,1],0,1]

    # create b by stacking x'_i and y'_i vertically
    b = np.zeros((X.shape[0]*2,1))
    for i in range(X.shape[0]):
        b[2*i] = X[i,2]
        b[2*i+1] = X[i,3]

    # fit a transformation y=Sx+t using numpy.linalg.lstsq 
    res, _, _, _ = np.linalg.lstsq(A,b,rcond=None)
    # res is 6x1, turn the first 4 elements into a 2x2 matrix S and the last 2 elements into a 2x1 matrix t
    S = np.array([[res[0],res[1]],[res[2],res[3]]])
    t = np.array([res[4],res[5]])

    # transform the points using np.dot
    y = np.dot(x,S) + t


    # Make as scatterplot of the points [xi,yi], [x′ i,y′ i] and S[xi,yi]T +t in one figure with different colors. 
    # Do this for both points_case_1.npy and point_case_2.npy. 
    # In other words, there should be two plots, each of which contains three sets of N points
    plt.scatter(X[:,0],X[:,1],color='r',label='x')
    plt.scatter(X[:,2],X[:,3],color='b',label='x\'')
    plt.scatter(y[:,0],y[:,1],color='g',label='Sx+t')
    plt.legend()
    plt.show()
   





def p4():
    # code for Task 4
    # load points X from task4/ points_case_5.npy and Y from task4/ points_case_9.npy [x1,y1,x1',y1']
    X = np.load('task4/points_case_9.npy')

    # fit a homography H1 and H2 using fit_homography for each set of points
    H1 = RANSAC_fit_homography(X)

    # transform the points using homography_transform
    y1 = homography_transform(X[:,:2],H1)

    # Visualize the original points [xi,yi], target points [x′ i,y′ i] and points after applying a homography transform T(H,[xi,yi]) in one figure. 
    plt.scatter(X[:,0],X[:,1],color='r',label='x')
    plt.scatter(X[:,2],X[:,3],color='b',label='x\'')
    plt.scatter(y1[:,0],y1[:,1],color='g',label='Hx')
    plt.legend()
    plt.show()


    



    


if __name__ == "__main__":
    # Task 3
    #p3()

    # Task 4
    p4()