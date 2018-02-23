# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---
[//]: # (Image References)

[hsv]: ./hsv.png "hsv"
[yelwhi]: ./yellow_white_lines.png "yellow_white_lines"
[gaus]: ./blur.png "gaus"
[canny]: ./canny.png "canny"

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
Then we transform the color selected frame to a grayscale frame, to make the edge detection step simpler
#### 3.Apply Gausseian blur:
To normalize any noise and sharpness, we perform a Gaussian blur on the grayscale image 
![alt text][gaus]
#### 4.Canny Edge Detection:
We then run the Canny Edge Detection algorithm to detect edges in the frame 
![alt text][canny]
#### 5.Region of Interest Selection:
We apply a Region of Interest mask which is a fixed polygon area to only retain the road and remove trees and fence from the frame 
#### 6.Hough Transform Line Detection:
Using probabilistic Hough transform we find line segments in the frame. Then, using draw_lines() function we draw the lines on the frame inplace. draw_lines() takes all the lines found by the Hough transform, splits them into left/right line segments using their slope, averages multiple segments to get a single line each for left and right lane and extrapolates the line from bottom of ROI to the top. The function also performs a moving average over previously found line segments to smooth out the lane lines drawn. 
#### 7.Merging Images:
now we merge our images.



---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

-the pipeline relies on a restricted region of interest based on several iteration.
-i belive the algorithm wont work as expected if the road was going up or down , as we have a constant egion of nterest.
-i have constant limits for color detections , i think the pipeline will perfom badly on darker or foggy road


### 3. Suggest possible improvements to your pipeline
-Dynamic selection of the region of interest.
-Identify a better method to detect curves.
-Identify better method to detect colors in differnt enviroment ,this should be dynamic


