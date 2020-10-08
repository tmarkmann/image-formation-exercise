import numpy as np
from cv2 import cv2
import glob

def showDistortedImages(pathToImages):
    images = glob.glob(pathToImages)
    print("Press any key to proceed to the next image")
    for imageName in images:
        image = cv2.imread(imageName)
        cv2.imshow('Distorted Image', image)
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

    for imageName in images:
        # Read chessboard images as black and white
        image = cv2.imread(imageName)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (boardColumns,boardRows), None)

        # If found, add object points and refined image points
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

    cv2.destroyAllWindows()
    imageSize = gray.shape[::-1]

    return worldPoints, imagePoints, imageSize


def undistortImage(worldPoints, imagePoints, imageSize, imagePath):
    ret, cameraMatrix, distortionParameter, rotationVectors, translationVectors = cv2.calibrateCamera(worldPoints, imagePoints, imageSize, None, None)

    image = cv2.imread(imagePath)
    height, width = image.shape[:2]
    imageSize = (width, height)

    # Get optimal new camera matrix
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionParameter, imageSize, 1, imageSize)

    # undistort image
    undistortedImage = cv2.undistort(image, cameraMatrix, distortionParameter, None, newCameraMatrix)
    
    return undistortedImage


def calculateReProjectionError(worldPoints, imagePoints, cameraMatrix, distortionParameter, rotationVectors, translationVectors):
    total_error = 0
    for i in range(len(worldPoints)):
        projectedImgPoints, _ = cv2.projectPoints(worldPoints[i], rotationVectors[i], translationVectors[i], cameraMatrix, distortionParameter)
        error = cv2.norm(imagePoints[i], projectedImgPoints, cv2.NORM_L2) / len(projectedImgPoints)
        total_error += error

    return total_error/len(worldPoints)


#def measureCoinSize(cameraMatrix, distortionParameter, rotationVectors, translationVectors):
#    pathToImage = 'res/measure/measure.jpg'
#    # Undistort image
#    undistortedImage = undistortImage(pathToImage, cameraMatrix, distortionParameter)
#
#    bbox = cv2.selectROI(undistortedImage, False)
#    print(bbox)



if __name__ == "__main__":
    chessboardSize = (7, 6)
    pathToImages = "res/chessboard/*.jpg"
    imageToUndistort = 'res/chessboard/left12.jpg'

    # Step 1: Comment
    #showDistortedImages(pathToImages)

    # Step 1: Uncomment
    worldPoints, imagePoints, imageSize = getChessboardPoints(pathToImages, chessboardSize, True)

    # Calculate Camera Parameter and undistort image
    # Step 2: Uncomment
    undistortedImage = undistortImage(worldPoints, imagePoints, imageSize, imageToUndistort)

    # Step 4
    # Calculate Re-Projection-Error
    #print(calculateReProjectionError(worldPoints, imagePoints, cameraMatrix, distortionParameter, rotationVectors, translationVectors))

    # Step 5
    #measureCoinSize(cameraMatrix, distortionParameter, rotationVectors, translationVectors)