'''There are various algorithm to compute a disparity map,
the one implemented in OpenCV is the graph cut algorithm.
'''
import cv2 as cv
import numpy as np

def cut(disparity, image, threshold):
 for i in range(0, image.height):
  for j in range(0, image.width):
   # keep closer object
   if cv.GetReal2D(disparity,i,j) > threshold: #Return a specific element of single-channel 1D, 2D, 3D or nD array.
    cv.Set2D(disparity,i,j,cv.Get2D(image,i,j)) #Change the particular array element.

# loading the stereo pair
left  = cv.imread('000000.png',cv.IMREAD_GRAYSCALE)
right = cv.imread('000001.png',cv.IMREAD_GRAYSCALE)

#create a matrix from each image
'''
disparity_left  = cv.CreateMat(left.height, left.width, cv.CV_16S)
disparity_right = cv.CreateMat(left.height, left.width, cv.CV_16S)
'''
disparity_left  = np.array((left.shape[0], left.shape[1]),np.uint8) #maybe changed to more approprate type
disparity_right = np.array((right.shape[0], right.shape[1]),np.uint8)

# Creates the state of graph cut-based stereo correspondence algorithm.
state = cv.CreateStereoGCState(16,2)
# Computes the disparity map using graph cut-based algorithm.
cv.FindStereoCorrespondenceGC(left,right,
                          disparity_left,disparity_right,state)

disp_left_visual = cv.CreateMat(left.height, left.width, cv.CV_8U)
cv.ConvertScale( disparity_left, disp_left_visual, -20 );
cv.Save( "disparity.pgm", disp_left_visual ); # save the map

# cutting the object farthest of a threshold (120)
cut(disp_left_visual,left,120)

cv.NamedWindow('Disparity map', cv.CV_WINDOW_AUTOSIZE)
cv.ShowImage('Disparity map', disp_left_visual)
cv.WaitKey()
