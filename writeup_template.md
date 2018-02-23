# **Finding Lane Lines on the Road** 


---
[//]: # (Image References)

[hsv]: ./writeup/hsv.png "hsv"
[yelwhi]: ./writeup/yellow_white_lines.png "yellow_white_lines"
[gaus]: ./writeup/blur.png "gaus"
[canny]: ./writeup/canny.png "canny"
[canny]: ./writeup/canny.png "canny"
[merged]: ./writeup/draw_lines_merge.png "merged"

# **Overview** 
the porppose of the project is to define a method to detect lanesof the road. When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. 
In this project you will detect lane lines in images using Python and OpenCV.


# **Proposed Pipeline**
The pipeline consists of the following steps
- Color Selection (Yellow and white lanes)
- Convert image to grayscale
- Canny Edge Detection
- Region of Interest Selection
- Hough Transform Line Detection

The above mentioned pipeline is using toprocess the video , the video is beeing looked at at singleimage where each mages goes throught he pipe line.

Below a detailed description of each step of the pipe line:
#### 1.Color selection:
We first run color selection on the input frame to select only the yellow and white shades of color and mask all other colors. To do this, we convert the input image to HSV (hue, saturation, value) color space. This makes it easier to select required colors using OpenCV's inRange function. 
![alt text][hsv]

Select yellow and white color

![alt text][yelwhi]

#### 2.Convert image to grayscale:
Then we transform the color selected frame to a grayscale frame to obtian better results for canny edge
To normalize any noise and sharpness, we perform a Gaussian blur on the grayscale image 
![alt text][gaus]
#### 4.Canny Edge Detection:
We then run the Canny Edge Detection algorithm to detect edges in the frames
![alt text][canny]
#### 5.Region of Interest Selection:
We apply a Region of Interest mask which is a fixed polygon area to only retain the road lanes.
#### 6.Hough Transform Line Detection:
Using probabilistic Hough transform we find line segments in the frame. 
Then, using draw_lines() function we draw the lines which represnts the road lanes. 
##### The function draw_lines() follows the below the steps to obtain/Draw the correct lines: 
- takes all the lines found by the Hough transform.
- filtering the lines and ignore verical lines , we conside them as noise.
- splits them into left/right line segments using their slope.
- find the interception , and ignore lines which deviate from the main min/max lines as we consider them the right and left lanes.
- averages multiple segments to get a single line each for left and right
#### 7.Merging Images:
now we merge our images, to obtain the final result
![alt text][merged]

---
###  Identify potential shortcomings with your current pipeline

- The pipeline relies on a restricted region of interest based on several iteration.
- I belive the algorithm wont work as expected if the road was going up or down , as we have a constant egion of nterest.
- I have constant limits for color detections , i think the pipeline will perfom badly on darker or foggy road


###  Suggest possible improvements to your pipeline
- Dynamic selection of the region of interest.
- Identify a better method to detect curves.
- Identify better method to detect colors in differnt enviroment ,this should be dynamic


