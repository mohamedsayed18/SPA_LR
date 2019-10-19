'''
stereo vision task: given the left and right image find
assumptions
baseline = 100mm
focal length = 2.8mm (for both cameras)
'''
#TODO the fundmental matrix
#import the modules
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('000000.png',0)   # left image
img2 = cv.imread('000001.png',0)   # right image
sift = cv.xfeatures2d.SIFT_create() #Scale Invariant Feature Transform

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# FLANN find match between two images
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
'''
This matcher trains cv::flann::Index on a train descriptor collection and calls
its nearest search methods to find the best matches.
So, this matcher may be faster when matching a large train collection than the brute force matcher. FlannBasedMatcher does not support masking permissible matches of descriptor sets
because flann::Index does not support this.
'''
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2) #this method find the match
good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS) #fundmental matrix
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#draw epilines of the points
#img1 - image on which we draw the epilines for the points in img2lines - corresponding epilines
def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist()) #choose a random color
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# drawing its lines on left image
'''
For points in an image of a stereo pair,
computes the corresponding epilines in the other image.
'''
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F) #the points, in which image, F = fundmentl matrix
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
'''
#TODO rectify images
#TODO find disparity map
#mn el a5er https://dzone.com/articles/computing-disparity-map-opencv
# There are various algorithm to compute a disparity map, the one implemented in OpenCV is the graph cut algorithm.
disparity = cv.StereoMatcher.compute()

 stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )
#imgL, imgR left and right camera
#https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
'''
