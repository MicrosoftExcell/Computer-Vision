#########################################################################
# Ensure file located in same directory as yolov3.cfg,yolov3.weights,coco.names and directory of images before running

# Loads, displays, computes SGBM disparity and detects and classifies
# objects for a set of rectified stereo images from a directory
# using the You Only Look Once object detection architecture
# Estimates object range by projecting the disparity to a 3D image and getting the minimum depth
# in a central portion of the detected object

# This code uses significant portions of code from:

# Title: stereo_display.py
# Author: Toby Breckon
# Date: 2017
# Availability: https://github.com/tobybreckon/stereo-disparity

# Title: yolo.py
# Author: Toby Breckon
# Date: 2019
# Availability: https://github.com/tobybreckon/python-examples-cv

# Title: stereo_to_3d.py
# Author: Toby Breckon
# Date: 2017
# Availability: https://github.com/tobybreckon/stereo-disparity

#########################################################################

# import necessary modules

import cv2
import os
import argparse
import sys
import math
import numpy as np
import csv

##########################################################################

#location of dataset
master_path_to_dataset = "TTBB-durham-02-10-17-sub10" # directory of images
directory_to_cycle_left = "left-images";    # edit this if needed
directory_to_cycle_right = "right-images";  # edit this if needed

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

pause_playback = False; # pause until key press after each image

##########################################################################

# parse command line arguments for YOLO files

parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-cl", "--class_file", type=str, help="list of classes", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str, help="network config", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str, help="network weights", default='yolov3.weights')

args = parser.parse_args()

#####################################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_centre_w = 474.5;

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

# uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);

# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

################################################################################
# dummy on trackbar callback function
def on_trackbar(val):
    return

###############################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, depth, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.1fm' % (class_name, depth)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (left, bottom + round(1.5*labelSize[1])),
        (left + round(1*labelSize[0]), bottom + round(0.5*baseLine)), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, bottom + round(1.25*labelSize[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

#####################################################################
# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)


################################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, max_disparity, rgb=[]):

    points = [];

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];
    
    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                Z = (f * B) / disparity[y,x];

                X = ((x - image_centre_w) * Z) / f;
                Y = ((y - image_centre_h) * Z) / f;

                # add to points

                # optional colour
                if(rgb.size > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
                else:
                    points.append([X,Y,Z]);

    return points;

#####################################################################

# project a set of 3D points back the 2D image domain

def project_3D_points_to_2D_image_points(points):

    points2 = [];

    for i1 in range(len(points)):

        # reverse earlier projection for X and Y to get x and y again

        x = ((points[i1][0] * camera_focal_length_px) / points[i1][2]) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / points[i1][2]) + image_centre_h;
        points2.append([x,y]);

    return points2;

################################################################################

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Load names of classes from file

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network using them

net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights_file)
output_layer_names = getOutputsNames(net)

 # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

#############################################################################

#iterate through files

for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the corresponding right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # check the file is a PNG file (left) and check a corresponding right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read frame
        frame = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        display_frame = frame

        # read right image
        # RGB image so load as such

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

#####################################################################################

        #apply pre-processing filters

        #conversion taken from [Mohammad Al Jazaery, 2016 - https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image]
        #equalising the histogram to account for changes in brightness in both images
        equ_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2YUV)
        equ_frame[:,:,0] = cv2.equalizeHist(equ_frame[:,:,0])
        equ_frame = cv2.cvtColor(equ_frame,cv2.COLOR_YUV2RGB)
        frame = equ_frame

        equ_imgR = cv2.cvtColor(imgR,cv2.COLOR_RGB2YUV)
        equ_imgR[:,:,0] = cv2.equalizeHist(equ_imgR[:,:,0])
        equ_imgR = cv2.cvtColor(equ_imgR,cv2.COLOR_YUV2RGB)
        imgR = equ_imgR

        # remove some Gaussian noise using gaussian filtering
        # apply Laplacian filtering for sharper edges
        
        smooth_frame = cv2.GaussianBlur(frame,(5,5),0)
        gray_frame = cv2.cvtColor(smooth_frame,cv2.COLOR_RGB2GRAY)
        lap_frame = cv2.Laplacian(gray_frame,cv2.CV_8U,3)
        lap_frame = cv2.cvtColor(lap_frame,cv2.COLOR_GRAY2RGB)
        frame = frame - lap_frame
        
        smooth_imgR = cv2.GaussianBlur(imgR,(5,5),0)
        gray_imgR = cv2.cvtColor(smooth_imgR,cv2.COLOR_RGB2GRAY)
        lap_imgR = cv2.Laplacian(gray_imgR,cv2.CV_8U,3)
        lap_imgR = cv2.cvtColor(lap_imgR,cv2.COLOR_GRAY2RGB)
        imgR = imgR - lap_imgR

################################################################################################

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        threshold = 30
        confThreshold = threshold/ 100
        classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');

        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)

        disparity = stereoProcessor.compute(grayL,grayR);

        # filter out noise and speckles (adjust parameters as needed)

        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        disparity_scaled = (disparity / 16.).astype(np.uint8);

        # project to a 3D colour point cloud (with or without colour)

        points = project_disparity_to_3d(disparity_scaled, max_disparity, frame);

        # project 3D points back to the 2D image

        pts = project_3D_points_to_2D_image_points(points);

###################################################################################
        # draw resulting detections and distances on image and calculate nearest object
        
        nearest = 1000
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            depth = 1000
            
            # control the boundaries of the object
            if left<0:
                left =0
            image_size = frame.shape
            if left+width > image_size[1]:
                width = image_size[1]-left

            # cutting out the left and the car bonnet
            # looking for the minimum depth in a box near the middle of the object
            # to avoid obscuring objects and pitfalls from just focusing on one point
            # Only looking at every 30 pixels to increase speed and still provide good estimation
            for i in range(0,len(pts),30):
                if pts[i][0]>70 and pts[i][1]<image_size[1]-400:
                    if round(pts[i][0])>round(left+(width/2.5)) and round(pts[i][0])<round(left+width-(width/2.5)) and round(pts[i][1])>round(top+(height/4)) and round(pts[i][1])<round(top+height-(height/3)):
                        new_depth = points[i][2]
                        if new_depth<depth:
                            depth = new_depth
                            point = pts[i]

            # if no depths found above
            
            if depth == 1000:
                
                # find the midpoint of the bounding box
                middle_x = round(left+(width/2))
                middle_y = round(top+(height/2))
                disparity_size = disparity_scaled.shape
                if middle_x<135: # ensure within bounds
                    middle_x = 135

                #get the depth at that pixel location
                for i in range(len(pts)):
                    if round(pts[i][0]) == middle_x and round(pts[i][1]) == middle_y:
                        depth = points[i][2]
                        point = pts[i]
                        break
                    elif round(pts[i][0]) == middle_x+1 and round(pts[i][1]) == middle_y:
                        depth = points[i+1][2]
                        point = pts[i]
                        break
                    elif round(pts[i][0]) == middle_x:
                        point = pts[i]
                        depth = points[i][2]
                        
                #searching until depth for object found
                if depth == 0:
                    for i in range(len(pts)):
                        if round(pts[i][1]) == middle_y:
                            point = pts[i]
                            depth = points[i][2]
                            break

            # object only acknowledged is within bounds for object detection
            # (missing the left, car bonnet and ignoring objects too far away to accurately get the distance of)
                    
            if top<300 and left>10 and (width>30 or height>width+10):
                
                # calculate nearest detected scene object
                if depth<nearest:
                    nearest = depth
            
                # drawing the point at which the depth was taken, the bounding box and the object's distance
                        
                #cv2.rectangle(display_frame, (int(point[0]), int(point[1])), (int(point[0])+1, int(point[1])+1), (0,0,255), 3)
                drawPred(display_frame, classes[classIDs[detected_object]],depth, left, top, left + width, top + height, (255, 178, 50))

##########################################################################################
        # output images, filenames and closest distance, show keys for actions
                
        # print filenames and nearest detected scene object
        print(filename_left)
        if nearest == 1000:
            nearest = 0
        print(filename_right,": nearest detected scene object ("+str("{:.1f}".format(nearest))+"m)")
        
        # display images (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)

        cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));
        cv2.imshow('left image',display_frame)
        
        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # pause - space

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled);
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);

# close all windows

cv2.destroyAllWindows()

#####################################################################

        
        

    
