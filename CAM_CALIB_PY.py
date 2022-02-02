import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.utils import shuffle
import cv2 as cv

def get_intrest_points():
    # WORLD CO-ORDINATES
    data = pd.read_excel("E:/GitHub/CV_Assignment_1/dataset.xlsx")
    x = np.array(data['X'])
    y = np.array(data['Y'])
    z = np.array(data['Z'])

    # IMAGE PIXEL COORDINATES
    u = np.array(data['x'])
    v = np.array(data['y'])

    # LIST OF WORLD PTS AND CORRESPONDING PIXEL PTS
    world_pts = []
    pixel_pts = []

    for i in range(len(x)):
        world_pts.append([x[i], y[i], z[i]])
        pixel_pts.append([u[i], v[i]])

    print("DATA FRAME")
    print(data)

    world_pts, pixel_pts = shuffle(world_pts, pixel_pts, random_state=0)

    return world_pts, pixel_pts

def RQ_decomposition(P):
    M = P[0:3,0:3]
    print("\nM:\n",M)
    K, R = linalg.rq(M)
    T = np.diag(np.sign(np.diag(K)))
    print(T)

    K = np.dot(K, T)
    R = np.dot(T, R)
    C = np.dot(linalg.inv(-M), P[:, 3])
    return K, R, C

def Normalization(nd, x):
    # CONVERTING TO NUMPY ARRAY
    x = np.asarray(x)
    # print(x)

    # CALCULATING centroid and mean distance from centroid
    m = np.mean(x, 0)
    dist = np.mean(np.sqrt(np.sum(np.square(x - m))))

    # NORMALIZATION MATRIX FOR WORLD POINTS(3D) AND IMAGE-PIXEL POINTS(2D)
    if nd == 2:
        s2D = np.sqrt(2) / dist
        Tr = np.diag([s2D, s2D, 1])
        Tr[0:2, 2] = -m * s2D

    else:
        s3D = np.sqrt(3) / dist
        Tr = np.diag([s3D, s3D, s3D, 1])
        Tr[0:3, 3] = -m * s3D

    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x
def Normalization(nd, x):
    # CONVERTING TO NUMPY ARRAY
    x = np.asarray(x)
    # print(x)

    # CALCULATING centroid and mean distance from centroid
    m = np.mean(x, 0)
    dist = np.mean(np.sqrt(np.sum(np.square(x - m))))

    # NORMALIZATION MATRIX FOR WORLD POINTS(3D) AND IMAGE-PIXEL POINTS(2D)
    if nd == 2:
        s2D = np.sqrt(2) / dist
        Tr = np.diag([s2D, s2D, 1])
        Tr[0:2, 2] = -m * s2D

    else:
        s3D = np.sqrt(3) / dist
        Tr = np.diag([s3D, s3D, s3D, 1])
        Tr[0:3, 3] = -m * s3D

    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    print("MEAN DISTANCE FROM CENTER : {}".format(np.mean(np.sqrt(np.sum(np.square(x))))))

    return Tr, x

def DLTcalib(nd, xyz, img_pt_):
    if (nd != 3):
        raise ValueError('%dD DLT unsupported.' % (nd))

    # Converting all variables to numpy array
    xyz = np.asarray(xyz)
    img_pt_ = np.asarray(img_pt_)

    n = xyz.shape[0]

    print("\n\nNUMBER OF POINTS: {}\n\n".format(n))
    # Validating the parameters:
    if img_pt_.shape[0] != n:
        raise ValueError(
            'Object (%d points) and image (%d points) have different number of points.' % (n, img_pt_.shape[0]))

    if (xyz.shape[1] != 3):
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' % (xyz.shape[1], nd, nd))

    if (n < 6):
        raise ValueError(
            '%dD DLT requires at least %d calibration points. Only %d points were entered.' % (nd, 2 * nd, n))

    Txyz, xyzn = Normalization(nd, xyz)
    Timg_pt_, img_pt_n = Normalization(2, img_pt_)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = img_pt_n[i, 0], img_pt_n[i, 1]
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

    # Convert A to array
    A = np.asarray(A)

    # Find the 11 parameters:
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]

    # Camera projection matrix
    H = L.reshape(3, nd + 1)

    # Denormalization
    H = np.dot(np.dot(np.linalg.pinv(Timg_pt_), H), Txyz)
    H = H / H[-1, -1]
    L = H.flatten()

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    img_pt_2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    img_pt_2 = img_pt_2 / img_pt_2[2, :]

    # Mean distance:
    projected_pts = img_pt_2.T
    actual_pixels = img_pt_

    # Printing teh actual and the estimated points
    print('\n\nACTUAL PTS   |             ESTIMATED PTS')
    for i in range(len(img_pt_2.T)):
        print("{}   <---->  {}".format(actual_pixels[i], projected_pts[i][0:2]))

    err = np.sqrt(np.mean(np.sum((img_pt_2[0:2, :].T - img_pt_) ** 2, 1)))

    return L, err, projected_pts[:,0:2]

def camera_param(P):
    print("\nP:\n",P)

    K, R, C = RQ_decomposition(P)

    print("\n\n K MATRIX: ")
    print(K)
    print("\n\n R MATRIX: ")
    print(R)
    print("\n\n Camera center: ")
    print(C)

    print("\n\n Normalized camera matrix: ")
    print(K/K[2][2])

def draw_on_image(original_pts, estimated_pts):
    img = cv.imread("E:\GitHub\CV_Assignment_1\THREE_PLANE_DATA\image (12).jpg")
    print("Success")
    cv.imshow('original image', img)
    cv.waitKey(1000)
    img_org = img.copy()
    img_est = img.copy()

    for i in original_pts:
        center = (i[0], i[1])
        radius = 5
        color_of_marker = (0,0,255)
        cv.circle(img_org, center, radius,color_of_marker,-1)
    cv.imshow('original image', img_org)
    cv.imwrite("ORIGINAL_INTEREST_POINTS.jpg", img_org)
    cv.waitKey(5000)

    for i in estimated_pts:
        center = (int(np.rint(i[0])), int(np.rint(i[1])))
        #print(center)
        radius = 5
        color_of_marker = (0,255,0)
        cv.circle(img_est, center, radius,color_of_marker, -1)
    cv.imshow('original image', img_est)
    cv.imwrite("ESTIMATED_POINTS.jpg", img_est)
    cv.waitKey(5000)

    for i in estimated_pts:
        center = (int(np.rint(i[0])), int(np.rint(i[1])))
        #print(center)
        radius = 5
        color_of_marker = (0,255,0)
        cv.circle(img_org, center, radius,color_of_marker, 2)
    cv.imshow('original image', img_org)
    cv.imwrite("ORIGINAL_ESTIMATED_POINTS_COMPARED.jpg", img_org)
    cv.waitKey(5000)


if __name__ == "__main__":
    xyz, img_pt_ = get_intrest_points()
    nd = 3
    P, err, estimated_pts = DLTcalib(nd, xyz, img_pt_)
    P = P.reshape(3,4)

    print('\n\nMatrix')
    print(P)

    print('\n\nError')
    print(err)

    print("\n\nCAMERA PARAMETERS: ")
    camera_param(P.reshape(3,4))

    draw_on_image(img_pt_,estimated_pts)