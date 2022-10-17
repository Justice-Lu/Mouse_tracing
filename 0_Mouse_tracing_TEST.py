# import the necessary packages
import argparse
import imutils
import cv2
import os
import sys 
from collections import deque
import numpy as np 
import pandas as pd
import datetime

import imageio


# construct arguments for the script 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the video file")
ap.add_argument("-o", "--output", default = '.',
    help="path to ouput csv data")
ap.add_argument("-m", "--mouse_name",
    help="mouse name for exported data")
ap.add_argument("-b", "--buffer", type=int, default=32,
    help="max buffer size")
ap.add_argument("-rm1", "--Room1_indent", type=int, default=0,
    help="indent x axis of room 1 by... ")
ap.add_argument("-rm2", "--Room2_indent", type=int, default=0,
    help="indent x axis of room 2 by...")
ap.add_argument("-sv", "--show_video", default=False,
    help="Show video tracing or not")
ap.add_argument("-sf", "--start_finding", type=int, default=10,
    help="Time to start finding mouse location")
args = vars(ap.parse_args())




# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
buffer = args["buffer"]
pts = deque(maxlen=buffer)
(dX, dY) = (0, 0)
direction = ""

# initialize the line and buffer that defines the rooms
left_buffer = args["Room1_indent"]
right_buffer = args["Room2_indent"]

video = cv2.VideoCapture(args['video'])
# object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20)
object_detector = cv2.createBackgroundSubtractorMOG2(varThreshold=20)


mouse_in_room = ""
center = None #location of detected motion 
mouse_location = None #Real location of mouse, Assigned after tracking for buffer length 
data = pd.DataFrame(columns=['mouse', 'time', 'position'])
if args["mouse_name"] == None:
    mouse_name = str(args['video'])



while True:
    # grab the current frame
    _, frame = video.read()
    if frame is None: 
        break 

    frame = imutils.resize(frame, width=500)
    height, width, _ = frame.shape
    

    # count the number of frames and track time 
    total_f = video.get(cv2.CAP_PROP_FRAME_COUNT)
    current_f = video.get(cv2.CAP_PROP_POS_FRAMES)
    fps = video.get(cv2.CAP_PROP_FPS)
    # calculate duration of the video    
    seconds = round(current_f / fps) 
    video_time = datetime.timedelta(seconds=seconds)   


    #Extract Region of interest
    roi = frame[round(height*3/5): round(height*4/5), round(width*1/10): round(width*9/10)]

    # Draws lines for marking the room    
    left_line_x = int(width*1/3)+left_buffer    
    right_line_x = int(width*2/3)+right_buffer
    cv2.line(frame, (left_line_x,0), (left_line_x,height), (0, 0, 0),1)
    cv2.line(frame, (right_line_x,0), (right_line_x,height), (0, 0, 0),1)

    #object detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 100, 255, 0)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select contours with the largest area, draw circle around it 
    if len(contours) > 0:
        cnt = max(contours, key= cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        M = cv2.moments(cnt)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if mouse_location is None: # if mouse location isn't assigned yet 
            if radius > 10: 
                cv2.circle(roi, (int(x), int(y)), int(radius), (0, 255, 0),2)
                pts.appendleft(center)
# If the detected center is >100 units away from mouse_location, then ignore it
        elif np.sqrt((mouse_location[0] - center[0])**2 + (mouse_location[1] - center[1])**2) < 100 and not mouse_location is None:
        	if radius > 10:
	        	cv2.circle(roi, (int(x), int(y)), int(radius), (0, 255, 0),2)
	        	pts.appendleft(center)
	        	mouse_location = center 
# Attempts to find mouse initial location after X seconds. 
        df = pd.DataFrame(pts)
        if seconds > args['start_finding'] and mouse_location is None:
        	if max(df[0]) - min(df[0]) > 100:
        		print('mice not found', (max(df[0]) - min(df[0])))
        	else:
        		mouse_location = (np.mean(df[0]), np.mean(df[1]))
        		print('mice found', mouse_location)

    # loop over the set of tracked points
    for i in np.arange(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # Draw the tracing line 
        thickness = int(np.sqrt(buffer/ float(i + 1)) * 2.5)
        cv2.line(roi, pts[i - 1], pts[i], (0, 255, 255), thickness)


        # Check which room mouse is in 
        if center[0] < left_line_x:
            mouse_in_room = "1st room"
        elif center[0] < right_line_x:
            mouse_in_room = "2nd room"
        else :
            mouse_in_room = "3rd room"

    # add text to video 
    cv2.putText(frame, mouse_in_room, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(frame, "video time: {}".format(video_time), (10,height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

    if args['show_video'] == "True":
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("roi", roi)
    
    key = cv2.waitKey(1)
    if key==ord("q"):
        break
        

    # Construct Datatable tracking the time and mouse information 
    if not mouse_location is None:
	    entry = pd.DataFrame({'mouse':[mouse_name], 'time':[seconds], 'position':[center]})
	    data = pd.concat([data, entry])

    # Time stamp on terminal to update progress 
    # print("Progress :" + str(current_f / total_f * 100))

video.release()
cv2.destroyAllWindows()


data.to_csv((args['output']+mouse_name+'.csv'))





