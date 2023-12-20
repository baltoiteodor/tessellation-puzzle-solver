import cv2 
import numpy as np 


# read original image 
img = cv2.imread("images/original_puzzle.jpg")

# convert original image to grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# save the greyscale image 
cv2.imwrite("images/grayscale_original_puzzle.jpg", gray)

# _, threshold_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
# threshold_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# threshold_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
threshold_img = cv2.Canny(img,100,200)

cv2.imwrite("images/threshold_original_puzzle.jpg", threshold_img)


# find all the shapes with the findContours method
contours, _ = cv2.findContours( 
    threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

# iterate thru contours 

first_shape = True  

for contour in contours: 
  
    # first shape is the whole image
    if first_shape: 
        first_shape = False
        continue
  
    # cv2.approxPloyDP() function to approximate the shape 
    approx = cv2.approxPolyDP( 
        contour, 0.01 * cv2.arcLength(contour, True), True) 
      
    # using drawContours() function 
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5) 
  
    # finding center point of shape 
    M = cv2.moments(contour) 
    if M['m00'] != 0.0: 
        x = int(M['m10']/M['m00']) 
        y = int(M['m01']/M['m00']) 
