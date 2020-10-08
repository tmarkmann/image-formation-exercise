# Image Formation - Camera Calibration and Measuring planar objects
This tutorial covers how to calibrate a camera using a chessboard pattern. You will learn about camera distortions, intrinsic and extrinsic parameters of a camera and how to find them. Using these parameters you will undistort images and measure the size of planar objects.  
This exercise and a majority of the following ones require programming skills in Python. If you are not familiar with Python or want to freshen up your Python skills you can find an extensive number of free tutorials online. Here are some examples:
* [Python for Beginners](https://www.python.org/about/gettingstarted/)  
* [Python Getting Started - w3schools](https://www.w3schools.com/python/python_getstarted.asp)  
* [Python Cheat Sheets](https://ehmatthes.github.io/pcc/cheatsheets/README.html)  
 
### OpenCV
You will also use openCV for some of the exercises. According to opencv.org "OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products."  
If you want to find out more about OpenCV please find further information here:
* [OpenCV](https://opencv.org/)  
* [OpenCV Introduction](https://docs.opencv.org/master/d1/dfb/intro.html)  

Information about the Python API used in this exercise can be found here:  
* [OpenCV on Wheels for Python](https://pypi.org/project/opencv-python/)  

## Setup
We strongly recommend you to use Ubuntu (18.04 or 20.04) for the exercises. If you want to setup the exercises on MacOS or Windows you can find plenty of tutorials online.  
1) First you need to install some packages: `sudo apt install python3 python3-pip python3-venv`
2) Open a terminal and clone the exercise repository: `git clone <TODO> && cd <TODO>`
3) Setup your Python environment by executing `virtualenv.sh`. This will create a virtual enviroment with all the python packages needed for the exercise
4) Activate your virtual environment:`source env/bin/activate`  
Now you're ready to work on the exercise :)

## Camera Distortions
Execute the python script with command: `python3 camera_calibration.py`. (***Hint:*** Press any key to proceed to the next picture)  
You should see a series of distorted images of chessboards. Due to camera distortion most straight line appear curved.  
<More about distortions? or just reference to lecture>

## Step 1 - Detect a chessboard pattern using openCV
The goal is to get rid of the camera distortion by calculating the distortion coefficients and a calibration matrix. To calculate them we need images of well defined patterns where we know the real world coordinates and the image coordinates. By using one of those well defined patterns, e.g. a chessboard, we can represent the search for the camera parameters as a mathematical problem. Luckily a bunch of smart people worked on that before us, so we don't have to deal with the mathematical details and can simply use a set of functions provided by openCV.  
Open the python script `camera_calibration.py` and look at function `getChessboardPoints(pathToImages, chessboardSize, display)`. This function reads in images at path 'pathToImages', tries to find chessboard corners using openCV function `cv2.findChessboardCorners().` and returns image coordinates with their corresponding world coordinates. Note that we can choose the world coordinates of the chessboard corners freely und therefore set them to their grid positions (i.e. (0,0), (1,0), (1,1) etc.) with z-coordinate '0'. Comment line <TODO>, uncomment line <TODO> and execute the python script again. You should see the detected chessboard corners.

## Step 2 - Calculate the camera parameters to undistort images
Now you can calculate the camera parameters with the openCV function `cv2.calibrateCamera()`. Uncomment line <TODO> and implement function undistortImage().  
Hints:  
* Use [cv2.calibrateCamera(worldPoints, imagePoints, imageSize, None, None)](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d) to get the camera parameters
* Use [cv2.imread(imagePath)](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) to read in the image to undistort
* Use [cv2.getOptimalNewCameraMatrix()](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1) to get an optimal camera matrix tuned for the scaling of the image to undistort
* Use [cv2.undistort()](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d) to undistort the image
* Use [cv2.imshow()](https://docs.opencv.org/4.4.0/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563) to view the undistorted image

## Step 3 - Calculate the Re-Projection Error
The Re-Projection Error measures how good the found camera parameters are. The world coordinates are projected to new image coordinates using the camera parameters. The Re-Projection Error is the mean of the euclidean distance between the real image coordinates and the projected ones. Uncomment line <TODO> and print the re-projection error. 

## Bonus Step - Measure planar object size
Print out the chessboard pattern provided under 'res/chessboard_print.pdf' and take photos (at least 10) from various angles with your smart phone.