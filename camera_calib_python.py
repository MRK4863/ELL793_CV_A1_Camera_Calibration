import numpy as np
import cv2 as cv
import glob
import pandas as pd
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def pretty_print(x):
    x_shape = x.shape
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            if (j==(x_shape[1]-1)):
                print("{:.4f}".format(x[i][j]), end = "")
            else:
                print("{:.4f} & ".format(x[i][j]), end = "")
        print('\\\\')

chessboard_size = (16,12)
frameSize = (1440, 1080)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_EPS, 30, 0.001)

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)


objPoints = []
imgPoints = []


images = glob.glob("E:\GitHub\CV_Assignment_1\TEST_DATA\*.jpg")
for image in images:
    print(image)
    img = cv.imread(image)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray_image", img_gray)
    # cv.waitKey(1000)

    #Find the chess board corners
    ret, corners = cv.findChessboardCorners(img_gray, chessboard_size, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    print("ret val = ", ret)
    
    if ret == True:
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)

        cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

    cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, frameSize, None, None)

print("Camera calibrated: ", ret)
print("\nCamera Matrix: \n", cameraMatrix)
pretty_print(cameraMatrix)

print("\nDistortion parameters: \n", dist)
pretty_print(dist)

print("\nRotation parameters: \n", rvecs)
print("\nTranslation parameters: \n", tvecs)


print('Normalized camera matrix:', cameraMatrix/cameraMatrix[2][2])


#ERROR

mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[1], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints2)
    mean_error +=error
    
print("\ntotal error: {}".format(mean_error/len(objPoints)))

print(len(objPoints))