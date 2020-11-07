import numpy as np
from cv2 import cv2
import math
import glob

def showDistortedImages(pathToImages):
    images = glob.glob(pathToImages)
    print("Press any key to proceed to the next image")
    for imageName in images:
        image = cv2.imread(imageName)
        imgheight, imgwidth = image.shape[:2]
        resizedImage = cv2.resize(image,(int(imgwidth/descalingFactor), int(imgheight/descalingFactor)), interpolation = cv2.INTER_AREA)
        cv2.imshow('Distorted Image', resizedImage)
        cv2.waitKey(0)

def getChessboardPoints(pathToImages, chessboardSize, display):
    boardColumns, boardRows = chessboardSize

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((boardRows*boardColumns,3), np.float32)
    objp[:,:2] = np.mgrid[0:boardColumns,0:boardRows].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    worldPoints = [] # 3d point in real world space
    imagePoints = [] # 2d points in image plane.

    # Get chessboard image file paths
    images = glob.glob(pathToImages)
    imageCounter = 1
    for imageName in images:
        print("Get chessboard pattern for image (", imageCounter, ",", len(images), ")")
        # Read chessboard images as black and white
        image = cv2.imread(imageName)
        imgheight, imgwidth = image.shape[:2]
        image = cv2.resize(image,(int(imgwidth/descalingFactor), int(imgheight/descalingFactor)), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (boardColumns,boardRows), None)
        if ret == True:
            worldPoints.append(objp)

            # Increase corner accuracy
            cornersRefined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imagePoints.append(cornersRefined)

            # Draw and display the corners if display was set to true 
            if display == True:
                image = cv2.drawChessboardCorners(image, (boardColumns,boardRows), cornersRefined,ret)
                cv2.imshow('img', image)
                cv2.waitKey(0)
        imageCounter += 1

    cv2.destroyAllWindows()
    imageSize = gray.shape[::-1]

    return worldPoints, imagePoints, imageSize


def undistortImage(worldPoints, imagePoints, imageSize, imagePath):
    ret, cameraMatrix, distortionParameter, rotationVectors, translationVectors = cv2.calibrateCamera(worldPoints, imagePoints, imageSize, None, None)

    # Read image, resize and convert to grey image
    image = cv2.imread(imagePath)
    height, width = image.shape[:2]
    image = cv2.resize(image,(int(width/descalingFactor), int(height/descalingFactor)), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    imageSize = (width, height)

    # Get optimal new camera matrix
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionParameter, imageSize, 1, imageSize)

    # undistort image
    undistortedImage = cv2.undistort(image, cameraMatrix, distortionParameter, None, newCameraMatrix)
    
    return undistortedImage, cameraMatrix, distortionParameter, rotationVectors, translationVectors


def calculateReProjectionError(worldPoints, imagePoints, cameraMatrix, distortionParameter, rotationVectors, translationVectors):
    total_error = 0
    for i in range(len(worldPoints)):
        projectedImgPoints, _ = cv2.projectPoints(worldPoints[i], rotationVectors[i], translationVectors[i], cameraMatrix, distortionParameter)
        error = cv2.norm(imagePoints[i], projectedImgPoints, cv2.NORM_L2) / len(projectedImgPoints)
        total_error += error

    return total_error/len(worldPoints)

def pointToWorld(cameraMatrix, rotationMatrix, translationVector, imagePoint):
    # image point as (x,y,1)
    ip = np.ones((3,1))
    ip[0,0] = imagePoint[0]
    ip[1,0] = imagePoint[1]
    ip = np.asmatrix(ip)
    # assumption: z-coordinate = 0
    z = 0
    # invert camera and rotation matrix
    r_inv = np.asmatrix(np.linalg.inv(rotationMatrix))
    c_inv = np.asmatrix(np.linalg.inv(cameraMatrix))
    # solve equation:
    # s * imagePoint = CameraMatrix * ( RotationMatrix * worldPoint + TranslationVector )
    # for worldPoint
    tempMat = r_inv * c_inv * ip
    tempMat2 = r_inv * translationVector
    s = (z + tempMat2[2,0]) / tempMat[2,0]
    worldPoint = s*tempMat - tempMat2

    return worldPoint

def measureDistanceSimple(image, measurePoints):
    boardColumns, boardRows = chessboardSize
    ret, corners = cv2.findChessboardCorners(image, (boardColumns,boardRows), None)
    if ret == True:
        print(corners)

def measureDistance(image, measurePoints, cameraMatrix, distortionParameter):
    # Get known world points
    boardColumns, boardRows = chessboardSize
    wPs = np.zeros((boardRows*boardColumns,3), np.float32)
    wPs[:,:2] = np.mgrid[0:boardColumns,0:boardRows].T.reshape(-1,2)
    wPs = wPs * 0.02

    ret, corners = cv2.findChessboardCorners(image, (boardColumns,boardRows), None)
    if ret == True:
        # Get known image points
        iPs = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Calculate rotation and translation vector
        _, rVec, tVec = cv2.solvePnP(wPs, iPs, cameraMatrix, distortionParameter)
        # Calculate rotation matrix
        r, _ = cv2.Rodrigues(rVec)        

        # Image points to world points
        p1 = pointToWorld(cameraMatrix, r, tVec, measurePoints[0])
        p2 = pointToWorld(cameraMatrix, r, tVec, measurePoints[1])

        # Euclidean distance
        distance = math.sqrt( (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 )
        return distance

def getMeasureCoord(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if(len(measurePoints) < 2):
            measurePoints.append([x,y])
            cv2.circle(undistortedImage, (x,y), 3, (0, 0, 255), thickness=3)
            cv2.imshow("Measure", undistortedImage)

if __name__ == "__main__":
    chessboardSize = (7, 9)
    descalingFactor = 5
    pathToImages = "res/own_chessboard/*.jpg"
    imageToUndistort = "res/own_chessboard/IMG_0694.jpg"

    # Step 1: Comment
    #showDistortedImages(pathToImages)

    # Step 1: Uncomment
    worldPoints, imagePoints, imageSize = getChessboardPoints(pathToImages, chessboardSize, False)

    # Calculate Camera Parameter and undistort image
    # Step 2: Uncomment
    undistortedImage = undistortImage(worldPoints, imagePoints, imageSize, imageToUndistort)
    cv2.imshow("Undistorted Image", undistortedImage)
    cv2.waitKey(0)

    # Step 4
    # Calculate Re-Projection-Error
    #print(calculateReProjectionError(worldPoints, imagePoints, cameraMatrix, distortionParameter, rotationVectors, translationVectors))

    # Step 5
    # Measuring planar objects
    imageToMeasure = "res/measure.jpg"
    undistortedImage, cameraMatrix, distCoeff, rVecs, tVecs = undistortImage(worldPoints, imagePoints, imageSize, imageToMeasure)
    measurePoints = []

    cv2.imshow("Measure", undistortedImage)
    cv2.setMouseCallback("Measure", getMeasureCoord)
    cv2.waitKey(0)

    print(measureDistanceSimple(undistortedImage, measurePoints))