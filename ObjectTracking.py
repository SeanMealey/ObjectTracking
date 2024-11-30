import sys
import os
import cv2
import numpy as np
import random
import math
from matplotlib import pyplot as plt



# Setup for object tracking

if not os.path.isdir(os.path.join(os.getcwd(), 'frames')):
    os.mkdir("frames")
else:
    print('frames already exists')

if not os.path.isdir(os.path.join(os.getcwd(), 'composite')):
    os.mkdir("composite")
else:
    print('composite already exists')
    
framenumber = 0
framectr = 0
pathToVideo = 'video2.mov'
omovie = cv2.VideoCapture(pathToVideo)
frame_height = omovie.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = omovie.get(cv2.CAP_PROP_FRAME_WIDTH)
firstFrameFlag = True

# Extract frames
while(1):
    ret, frame = omovie.read()
    if not ret:
        break
    print('Extracting: %d' % framenumber)
    cv2.imwrite('frames/%d.tif' % framenumber, frame)
    framenumber += 1
omovie.release()

def smooth_coordinates(coords, window_size=5):
    """
    Apply moving average smoothing to a list of coordinates
    """
    if len(coords) < window_size:
        return coords
    
    smoothed = []
    for i in range(len(coords)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(coords), i + window_size // 2 + 1)
        window = coords[start_idx:end_idx]
        smoothed.append(sum(window) / len(window))
    return smoothed

# Find object by averaging the foreground coordinates
def findObj(frame, threshold):
    count = 1
    xcount = 0
    ycount = 0
    for row in range(frame.shape[0]):
        for column in range(frame.shape[1]):
            if frame[row][column][0]>250 and frame[row][column][1]>250 and frame[row][column][2]>250:
                count += 1
                xcount += row
                ycount += column
    averagex = xcount/count
    averagey = ycount/count
    radius = math.sqrt(count/math.pi)
    coord = [averagex,averagey]
    return coord

def isObjectThere(frame, threshold):
    count = 1
    xcount = 0
    ycount = 0
    for row in range(frame.shape[0]):
        for column in range(frame.shape[1]):
            if frame[row][column][0]>250 and frame[row][column][1]>250 and frame[row][column][2]>250:
                count += 1
                xcount += row
                ycount += column
    averagex = xcount/count
    averagey = ycount/count
    radius = math.sqrt(count/math.pi)
    coord = [averagex,averagey]
    if count>threshold:
        return True
    else:
        return False

def drawbox(frame, centerx, centery, radius, color):
    for y in range(centerx - radius, centerx + radius):
        for x in range(centery - radius, centery + radius):
            cx = 0 if x < 0 else frame.shape[0]-1 if x > frame.shape[0] - 1 else x
            cy = 0 if y < 0 else frame.shape[1]-1 if y > frame.shape[1] - 1 else y
            for i in range(3):
                frame[cx][cy][i] = color[i]
    return frame

framectr = framenumber - 1
process_frame = 0

foreground = 250 # Foreground Threshold for Segmentation


coordListX = list()
coordListY = list()
coordList = list()
count = 0
presentFrames = []
presentFrameCount = 0

process_frame = 0

while process_frame <= framectr:
    oframe = cv2.imread('frames/%d.tif' % process_frame)
    print('Processing frame: %d, overall progress: %.2f %%' % (process_frame, process_frame/framectr*100))
    
    # Change frame to grey scale
    gframe = oframe.copy() 

    # Load the saved frames sequentially
    height = None
    width = None
    for y in range(gframe.shape[1]):
        for x in range(gframe.shape[0]):
            # Convert to gray scale
            g = 0.212671 * gframe[x][y][2] + 0.715160 * gframe[x][y][1] + 0.072169 * gframe[x][y][0]

            # Convert to binary
            for i in range(3):
                if g > foreground:
                    gframe[x][y][i] = 255
                else:
                    gframe[x][y][i] = 0

    # Get the initial object coordinates
    coord = findObj(gframe, 128) 
            
    # Draw red dot in the center of the  object
    threshold = 250
    combined_img = np.hstack((oframe, gframe))
    if isObjectThere(oframe, threshold):
        if not firstFrameFlag:
            oframe = drawbox(oframe, int(coord[1]), int(coord[0]), 5, (0, 0, 255))
            gframe = drawbox(gframe, int(coord[1]), int(coord[0]), 5, (0, 0, 255))
            combined_img = np.hstack((oframe, gframe))
        firstFrameFlag = False
    coordListX.append(coord[1])
    coordListY.append(coord[0])
    

    
    if isObjectThere(gframe, 128):
        gframe = cv2.putText(combined_img, text='Object found!', org=(2200, 1000),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=5, thickness = 5, 
color=(0, 255, 0))
        firstFrameFlag = True
        presentFrames.append(combined_img)
        x1 = int(coordListX[count-1])
        y1 = int(coordListY[count-1])
        x2 = int(coordListX[count])
        y2 = int(coordListY[count])
        point1 = (x1, y1)
        coordList.append(point1)
        point2 = (x2, y2)
        print(presentFrameCount)
        presentFrameCount += 1
    else:
        gframe = cv2.putText(combined_img, text='Object is missing!', org=(2200, 1000),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=5, thickness = 5, 
color=(0, 255, 0))
        
    
    if presentFrameCount > 1:
        # Smooth the x and y coordinates separately
        smooth_x = smooth_coordinates([coord[0] for coord in coordList])
        smooth_y = smooth_coordinates([coord[1] for coord in coordList])
        
        # Update coordList with smoothed coordinates
        coordList = list(zip(map(int, smooth_x), map(int, smooth_y)))

        # Draw the smoothed path
        for i in range(1, presentFrameCount):    
            combined_img = cv2.line(combined_img, coordList[i-1], coordList[i], (0, 0, 255), 5)
            cv2.imwrite('composite/composite%d.tif' % process_frame, combined_img)
    
    for i in range(1,presentFrameCount):    
        combined_img = cv2.line(combined_img, coordList[i-1], coordList[i], (0, 0, 255), 5) 
        cv2.imwrite('composite/composite%d.tif' % process_frame, combined_img)
                         
    
    cv2.imwrite('composite/composite%d.tif' % process_frame, combined_img)
    
    
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
    process_frame += 1
    count += 1


count = 0
out = cv2.VideoWriter('./NewVideo.mov', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (int(frame_width*2), int(frame_height)))
while(1):
    img = cv2.imread('composite/composite%d.tif' % count)
    if img is None:
        print('No more frames to be loaded')
        break
    out.write(img)
    count += 1
    print('Saving video: %d%%' % int(100*count/framenumber))
    
out.release()
cv2.destroyAllWindows()

path_to_output = './NewVideo.mov'
