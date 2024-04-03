import cv2 as cv
import numpy as np

def Edge_detection(frame):
    #image read and show
    frame = cv.imread('house.jpg')
    cv.imshow('your given img',frame)

    #concertion rgb image to gray scale image
    gray_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('gray img', gray_img)

    #blur images using gaussian kernal 
    blur_img = cv.GaussianBlur(gray_img,(3,3),cv.BORDER_DEFAULT)
    cv.imshow('blur img', blur_img)

#blur images using gaussian kernal (3,3)
# blur_img3 = cv.GaussianBlur(gray_img,(3,3),cv.BORDER_DEFAULT)
# cv.imshow('blur img 3 3', blur_img3)

    #Edge cascade
    canny_img = cv.Canny(blur_img,125,275)
    cv.imshow('canny img, final img', canny_img)

    #Dilating the image
    dilate_img = cv.dilate(canny_img, (5,5), iterations=1)
    cv.imshow('Dilate img', dilate_img)

    #eroding image
    erode_img = cv.erode(dilate_img, (3,3), iterations=1)
    cv.imshow('Erode img',erode_img)
    

    #laplacian opearator for edge detection 
    lap = cv.Laplacian(blur_img, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    cv.imshow('laplacian edge detect', lap)

    #sobel operation for edge detection
    sobel_x = cv.Sobel(blur_img, cv.CV_64F, 1, 0)
    sobel_y = cv.Sobel(blur_img, cv.CV_64F, 0, 1)
    combine_sobel = cv.bitwise_or(sobel_x, sobel_y)
    cv.imshow('edge detion by sobel operator', combine_sobel)

#Edge cascade from 3,3
# canny_img3 = cv.Canny(blur_img3,125,175)
# cv.imshow('canny img3', canny_img3)



Edge_detection('house.jpg')

cv.waitKey(0)
cv.destroyAllWindows()