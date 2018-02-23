#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from scipy import stats

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.


debug = 1
count =0

#helper functions defintions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def get_lines_slope_intercepts(lines):
	 #create a list of slope and intercepts for all lines
    var_slope_intercept = np.zeros((len(lines),2))
    #calculate the slope for the lines we have
    #print (var_slope_intercept)
    for i,line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            if x2==x1:
                continue # ignore a vertical line
        slope = (y2-y1)/(x2-x1)
        intercept = y1 - x1 * slope
        var_slope_intercept[i]=[slope,intercept]
    return (var_slope_intercept)


def get_lines_in_range(xf_size,var_slope_intercept_p,left_slope_max,right_slope_min):
    #get the max , min slope:
    devaitation_slope = 0.050
    deviation_x_intercept = 0.050 * xf_size
    left_slopes = []
    left_intercepts = []
    right_slopes = []
    right_intercepts = []
    # this gets slopes and intercepts of lines similar to the lines with the max (immediate left) and min
    # (immediate right) slopes (i.e. slope and intercept within x%)
    for var_line_slopes_intercepts_matrix in var_slope_intercept_p:
        # print (0.15 * x_size)
        # print (var_line_slopes_intercepts_matrix[0] - max_slope_line[0])
        # print (var_line_slopes_intercepts_matrix[1] - max_slope_line[1])
        if (abs(var_line_slopes_intercepts_matrix[0] - left_slope_max[0]) < devaitation_slope) and (abs(var_line_slopes_intercepts_matrix[1] - left_slope_max[1]) < (deviation_x_intercept)):
            right_slopes.append(var_line_slopes_intercepts_matrix[0])
            right_intercepts.append(var_line_slopes_intercepts_matrix[1])
        elif (abs(var_line_slopes_intercepts_matrix[0] - right_slope_min[0]) < devaitation_slope) and (abs(var_line_slopes_intercepts_matrix[1] - right_slope_min[1]) < (deviation_x_intercept)):
            left_slopes.append(var_line_slopes_intercepts_matrix[0])
            left_intercepts.append(var_line_slopes_intercepts_matrix[1])
    return (right_slopes,right_intercepts,left_slopes,left_intercepts)


def draw_lines(img, lines, color_l=[255, 0, 0],color_r=[0, 255, 0] , thickness=10):
    x_size = img.shape[1]
    y_size = img.shape[0]
    left_slopes = []
    left_intercepts = []
    right_slopes = []
    right_intercepts = []

    var_slope_intercept=get_lines_slope_intercepts(lines)

    #get max/ min slope of the detected lines
    max_slope_line = var_slope_intercept[var_slope_intercept.argmax(axis=0)[0]]
    min_slope_line = var_slope_intercept[var_slope_intercept.argmin(axis=0)[0]]

    (right_slopes,right_intercepts ,left_slopes , left_intercepts) = get_lines_in_range(x_size,var_slope_intercept,max_slope_line,min_slope_line)

    # left and right lines are averages of these slopes and intercepts, extrapolate lines to edges and center*
    # *roughly

    left_avergaed_lines= np.zeros(shape=(1,1,4), dtype=np.int32)
    right_avergaed_lines= np.zeros(shape=(1,1,4), dtype=np.int32)
    #iterate over the left slops lines
    if len(left_slopes) > 0:
        left_line = [sum(left_slopes)/len(left_slopes),sum(left_intercepts)/len(left_intercepts)]
        left_line_x = (y_size - left_line[1])/left_line[0]
        left_top_x = (y_size*.6 - left_line[1])/left_line[0]
        if (left_line_x > 0):
            left_avergaed_lines[0][0] =[left_line_x,y_size,left_top_x,y_size*.6]

#iterate over the rigjht slops lines

    if len(right_slopes) > 0:
        right_line = [sum(right_slopes)/len(right_slopes),sum(right_intercepts)/len(right_intercepts)]
        right_bottom_x = (y_size - right_line[1])/right_line[0]
        right_top_x = (y_size*.6 - right_line[1])/right_line[0]
        if (right_bottom_x < x_size):
            right_avergaed_lines[0][0]=[right_bottom_x,y_size,right_top_x,y_size*.6]

#actively drawing the line
    for line in left_avergaed_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color_l, thickness)
#actively drawing the line
    for line in right_avergaed_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color_r, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=1, β=0.95, γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

#a debug function i use it to monitor my pipeline excution
def print_gimage(img, title = "default"):
    if (debug == 12):
        print(title);
        plt.imshow(img,cmap='gray')
        plt.show()


#end function definition s
def proces_image(image):
    #print("test_images/"+img)
    image_o = np.copy(image)
    #print_gimage(image)
    gray_img = grayscale(image)
    print_gimage(image,"orignial image")

    hsv_img = hsv(image_o)
    # define range of color in HSV, Yellow and white
    lower_yel = np.array([20,100,100])
    upper_yel = np.array([30,255,255])
    lower_wht = np.array([0,0,235])
    upper_wht = np.array([255,255,255])

    #debug code
    print_gimage(hsv_img,"hsv_img")


    # Threshold the HSV image to get only yellow/white
    yellow_mask = cv2.inRange(hsv_img, lower_yel, upper_yel)
    white_mask = cv2.inRange(hsv_img, lower_wht, upper_wht)

    #debug code
    print_gimage(yellow_mask,"yellow_mask")
    print_gimage(white_mask,"white_mask")

    # Bitwise-AND mask and original image
    full_mask = cv2.bitwise_or(yellow_mask, white_mask)

    subdued_gray = (gray_img / 2).astype('uint8')
    print_gimage(subdued_gray,"subdued_gray")
    print_gimage(full_mask,"full_mask")
    boosted_lanes = cv2.bitwise_or(subdued_gray, full_mask)

    print_gimage(boosted_lanes,"boosted_lanes")

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 15
    image = gaussian_blur(boosted_lanes,kernel_size)
    print_gimage(image,"beforecanny")

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    image = canny(image,low_threshold,high_threshold)
    print_gimage(image,"canny")


    imshape = image.shape
    x = image.shape[1]
    y = image.shape[0]
    #print (str(x) + " , " + str(y))
    vertices = np.array([[(x*0.,y),(x*.475, y*.575), (x*.525, y*.575), (x,y)]], dtype=np.int32)
    #print (vertices)
    image = region_of_interest(image,vertices)
    print_gimage(image)
    line_image = np.copy(image)*0 # creating a blank to draw lines on


    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  #minimum number of pixels making up a line
    max_line_gap =mg, rho, theta, threshold, min_line_len, max_line_gap):
    image = hough_lines(image,rho,theta,threshold,min_line_length,max_line_gap)


    result = weighted_img(image,image_o)
    return result
 300    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    #hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    image = hough_lines(image,rho,theta,threshold,min_line_length,max_line_gap)


    result = weighted_img(image,image_o)
    return result

#process all the images in a folder.
def loop_proces():
    for img_n in os.listdir("test_images/"):
        image = mpimg.imread("test_images/"+img_n)
        print_gimage(proces_image(image))
        break

#a function to loop around all the videos in a folder and process them
def process_video():
    global count
    for vid_n in os.listdir("test_videos/"):
        white_output = 'test_videos_output/'+vid_n
        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
        clip1 = VideoFileClip("test_videos/"+vid_n)
        white_clip = clip1.fl_image(proces_image) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)


#step one is color selection ,i need to extract colors of interest for the lane:
#loop_proces()
process_video()
