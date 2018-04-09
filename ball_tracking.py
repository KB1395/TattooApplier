# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

#load tatoo image
imgTatoo=cv2.imread('mustache.png',-1)
tatMask=imgTatoo[:,:,3]
#create a mask from the image
invTatMask=cv2.bitwise_not(tatMask)
imgTatoo=imgTatoo[:,:,0:3]
#define original sizes for the tatoo
tatOrigHeight,tatOrigWidth = imgTatoo.shape[:2]


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (0, 0, 73)
greenUpper = (35, 93, 255)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	frame = cv2.bilateralFilter(frame, 11, 17, 17)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	
	

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 10:
			#draw contour of desired shape
			cv2.drawContours( frame, c, -1, (239, 0, 0),6 )
			#create the smallest box containing that contour
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			#draw the box
			cv2.drawContours(frame,[box],0,(0,0,255),2)
			#Save the box parameters (center,height,width and angle)
			areaCenter=rect[0]
			areaX,areaY=int(areaCenter[0]),int(areaCenter[1])
			areaSize=rect[1]
			areaHeight=int(areaSize[0])
			areaWidth=int(areaSize[1])
			areaAngle=rect[2]
			#define the tattoo size
			tatWidth=int(0.2*areaWidth)
			tatHeight=tatWidth * tatOrigHeight // tatOrigWidth
			
			#face = cv2.rectangle(frame,(areaX-areaWidth//4,areaY-areaHeight//4),(areaX+areaWidth//4,areaY+areaHeight//4),(255,0,0),2)
			
			#roiGray=gray[areaY-areaHeight//2:areaY+areaHeight//2, areaX-areaWidth//2:areaX+areaWidth//2]
			
			#create a mask from the video feed with the size of the region of interest (box created before)
			roiColor=frame[areaY-areaHeight//2:areaY+areaHeight//2, areaX-areaWidth//2:areaX+areaWidth//2]
			
			# print(areaX,areaY,areaWidth,areaHeight)
			# print(tatWidth,tatHeight)
			
			# save the center of the region of interest (ROI)
			x1 = areaX - (tatWidth//2)
			x2 = areaX + (tatWidth//2)
			y1 = areaY - (tatHeight//2)
			y2 = areaY + (tatHeight//2)
			
			# protect from wierd center coordinates (outside of the frame)
			if x1 < 0:
				x1 = 0
			if y1 < 0:
				y1 = 0
			if x2 > areaWidth:
				x2 = areaWidth
			if y2 > areaHeight:
				y2 = areaHeight
			print(x1,x2,y1,y2)
			
			# resize the tattoo to match the ROI size
			tatHeight=tatWidth * tatOrigHeight // tatOrigWidth
			tatWidth=x2-x1
			
			# protect from wierd (negative) tatoo sizes
			if tatHeight<=0:
				tatHeight=1
			if tatWidth<=0:
				tatWidth=2
			print(tatHeight)
			print(tatWidth)
			
			# resize all the masks to the same size in order to merge them
			tatoo=cv2.resize(imgTatoo,(tatWidth,tatHeight),interpolation=cv2.INTER_AREA)
			mask2=cv2.resize(tatMask,(tatWidth,tatHeight),interpolation=cv2.INTER_AREA)
			mask2inv=cv2.resize(invTatMask,(tatWidth,tatHeight),interpolation=cv2.INTER_AREA)
			
			
			print(mask2inv.shape)
			
			#attempt to save the ROI coordinates
			fy1=y1
			fy2=y1+tatHeight
			fx1=x1
			fx2=x1+tatWidth
			
			#create a ROI mask
			roi = frame[fy1:fy2,fx1:fx2]
			print(roi.shape)
			
            #merge the roi mask with the tatoo and the inverted tatoo masks
			roi_bg = cv2.bitwise_and(roi,roi,mask = mask2inv)
			roi_fg = cv2.bitwise_and(tatoo,tatoo,mask = mask2)
			
			print(roi_bg.shape,roi_fg.shape)
			
			#merge the background and foreground ROI masks
			dst = cv2.add(roi_bg,roi_fg)
			
			print("dst: ",dst.shape)
			print("roi: ",roiColor.shape)
			print(fy1,fy2,fy2-fy1)
			print(fx1,fx2,fx2-fx1)
			
			# add the merged mask to the video feed
			roiColor[fy1:fy2,fx1:fx2]=dst
		

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()