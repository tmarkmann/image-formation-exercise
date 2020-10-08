import numpy as np
from cv2 import cv2
import glob

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
        image = cv2.resize(image, (604, 806), image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (boardColumns,boardRows), None)
        # If found, add object points and refined image points
        if ret == True:
            worldPoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imagePoints.append(corners2)

            # Draw and display the corners if display was set to true 
            if display == True:
                image = cv2.drawChessboardCorners(image, (boardColumns,boardRows), corners2,ret)
                cv2.imshow('img', image)
                cv2.waitKey(0)

    cv2.destroyAllWindows()
    imageSize = gray.shape[::-1]

    return worldPoints, imagePoints, imageSize


def undistortImage(imagePath, cameraMatrix, distortionParameter):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (604, 806), image)
    height, width = image.shape[:2]
    imageSize = (width, height)

    # Get optimal new camera matrix
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionParameter, imageSize, 1, imageSize)

    # undistort image
    undistortedImage = cv2.undistort(image, cameraMatrix, distortionParameter, None, newCameraMatrix)
    
    # crop the image
    x,y,w,h = roi
    undistortedImage = undistortedImage[y:y+h, x:x+w]
    return undistortedImage


def calculateReProjectionError(worldPoints, imagePoints, cameraMatrix, distortionParameter, rotationVectors, translationVectors):
    total_error = 0
    for i in range(len(worldPoints)):
        projectedImgPoints, _ = cv2.projectPoints(worldPoints[i], rotationVectors[i], translationVectors[i], cameraMatrix, distortionParameter)
        error = cv2.norm(imagePoints[i], projectedImgPoints, cv2.NORM_L2) / len(projectedImgPoints)
        total_error += error

    return total_error/len(worldPoints)


def measureCoinSize(cameraMatrix, distortionParameter, rotationVectors, translationVectors):
    pathToImage = 'res/measure/measure.jpg'
    # Undistort image
    undistortedImage = undistortImage(pathToImage, cameraMatrix, distortionParameter)

    bbox = cv2.selectROI(undistortedImage, False)
    print(bbox)



if __name__ == "__main__":
    cheassboardSize = (7, 9)
    #cheassboardSize = (7, 6)
    pathToImages = "res/own_chessboard/*.jpg"
    #pathToImages = "res/chessboard/*.jpg"
    imageToUndistort = 'res/own_chessboard/IMG_0692.jpg'
    #imageToUndistort = 'res/chessboard/left12.jpg'

    # Get Chessboard points
    worldPoints, imagePoints, imageSize = getChessboardPoints(pathToImages, cheassboardSize, False)

    # Calculate Camera Parameter
    ret, cameraMatrix, distortionParameter, rotationVectors, translationVectors = cv2.calibrateCamera(worldPoints, imagePoints, imageSize, None, None)
    print(cameraMatrix)

    # Undistort image
    undistortedImage = undistortImage(imageToUndistort, cameraMatrix, distortionParameter)
    cv2.imwrite('calibrationRresult.png', undistortedImage)

    # Calculate Re-Projection-Error
    print(calculateReProjectionError(worldPoints, imagePoints, cameraMatrix, distortionParameter, rotationVectors, translationVectors))

    measureCoinSize(cameraMatrix, distortionParameter, rotationVectors, translationVectors)
