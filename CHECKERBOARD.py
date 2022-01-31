import numpy as np
import cv2 as cv
import glob

chessboard_size = [(i,j) for i in range(12,20,1) for j in range(10, 20, 1)]

print(chessboard_size)

frameSize = (1440, 1080)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_EPS, 30, 0.001)

# objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
#
# objPoints = []
# imgPoints = []

image = "E:\GitHub\CV_Assignment_1\TEST_IMAGES_3\image (1).jpg"

print(image)
img = cv.imread(image)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("gray_image", img_gray)
# cv.waitKey(1000)

# Find the chess board corners
for i in chessboard_size:
    print(i)
    ret, corners = cv.findChessboardCorners(img_gray, i, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    print("ret val = ", ret)

