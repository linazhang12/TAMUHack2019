import os
import sys
import cv2
import numpy as np

inputVideo = cv2.VideoCapture(0)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

while (True):
	# capture frame-by-frame
	ret, image = inputVideo.read() 
	image = cv2.imread('image.png',0)

	image = np.array(image, dtype = np.uint8)
	imageCopy = image

	# https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python
	# adjust hue, saturation, and vibrance     h,s,v
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv)
	h += 0
	s += 0
	v += 0
	final_hsv = cv2.merge((h,s,v))
	imageCopy = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

	# adjust brightness and contrast
	brightness = 0
	contrast = 0
	imageCopy = np.int16(imageCopy)
	imageCopy = imageCopy * (contrast/127+1) - contrast + brightness
	imageCopy = np.clip(imageCopy, 0, 255)
	imageCopy = np.uint8(imageCopy)
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# imageCopy = gray
	# h, s, v = cv2.split(gray)
	# h += 60
	# s += 0 
	# v += 0
	# imageCopy = cv2.merge((h,s,v))
	# imageCopy = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


	corners, ids, rejectedImgPoints = (cv2.aruco.detectMarkers(image,dictionary))

	if (len(corners) == 1):
		imageCopy = cv2.aruco.drawDetectedMarkers(image,corners,ids)
		
	cv2.imshow("out",imageCopy)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
inputVideo.release()
cv2.destroyAllWindows()