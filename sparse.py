#########################################################################
# Ensure file located in same directory as yolov3.cfg,yolov3.weights,coco.names and directory of images before running

# Loads, displays images from directory
# Detects and classifies objects for a set of rectified stereo images from a directory
# using the You Only Look Once object detection architecture
# Estimates the object range by finding the depth of the feature point closest to the object
# (using ORB) or using linear interpolation to estimate the distance to a point

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

# Title: surf_detection.py
# Author: Toby Breckon
# Date: 2016/17
# Availability: https://github.com/tobybreckon/python-examples-cv

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

interpolate = True; # True to use linear interpolation, False to find the nearest feature point (faster but less accurate)

#location of dataset
master_path_to_dataset = "TTBB-durham-02-10-17-sub10"
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

## project_disparity_to_3d : project feature points to a set of 3D points

def project_disparity_to_3d(img1_coords,img2_coords):

    points = [];

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;
    
    for i in range(len(img1_coords)):

        #getting each feature point
        (x1,y1) = img1_coords[i]
        (x2,y2) = img2_coords[i]

        #calculating the disparity between the feature points
        disparity = abs(x1-x2)

        if (disparity > 0):

            # calculate corresponding 3D point [X, Y, Z]

            Z = (f * B) / disparity;

            #X = ((x1 - image_centre_w) * Z) / f;
            #Y = ((y1 - image_centre_h) * Z) / f;

            # add original coords and depth to points

            points.append([x1,y1,Z]);

    return points;

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
        
        # using ORB (with Max Features = 800) - [Rublee et al., 2011 - https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF]

        feature_object = cv2.ORB_create(800)
        
        # use FLANN object that can handle binary descriptors
        # taken from: https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
        # N.B. "commented values are recommended as per the docs,
        # but it didn't provide required results in some cases"

        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6, # 12
                        key_size = 12,     # 20
                        multi_probe_level = 1) #2

        # create a Fast Linear Approx. Nearest Neightbours (Kd-tree) object for
        # fast feature matching
        # ^ ^ ^ ^ yes - in an ideal world, but in the world where this issue
        # still remains open in OpenCV 3.1 (https://github.com/opencv/opencv/issues/5667)
        # just use the slower Brute Force matcher and go to bed
        # summary: python OpenCV bindings issue, ok to use in C++ or OpenCV > 3.1

        (major, minor, _) = cv2.__version__.split(".")
        if ((int(major) >= 3) and (int(minor) >= 1)):
            search_params = dict(checks=50)   # or pass empty dictionary
            matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else:
            matcher = cv2.BFMatcher()
                
        # read frame
        frame = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        display_frame = frame

        # read right image
        # RGB image so load as such

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

#######################################################################################

        #apply pre-processing filters

        #conversion used from [Mohammad Al Jazaery, 2016 - https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image]
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

#########################################################################################

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

#########################################################################################
        #get feature points and depths
        
        h, w, c = frame.shape   # size of left image
        cropped = False
        if (h>0) and (w>0):
            cropped = True
            # detect features and compute associated descriptor vectors

            keypoints_region, descriptors_region = feature_object.detectAndCompute(frame,None)

        if (cropped):

            # detect and match features from right image

            keypoints, descriptors = feature_object.detectAndCompute(imgR,None)

            matches = []
            if (len(descriptors) > 0):
                matches = matcher.knnMatch(descriptors_region, trainDescriptors = descriptors, k = 2)

            # Need to isolate only good matches
            # perform a first match to second match ratio test as original SIFT paper (known as Lowe's ration)
            # using the matching distances of the first and second matches

            good_matches = []
            try:
                for (m,n) in matches:
                    if m.distance < 0.7*n.distance:
                        good_matches.append(m)
            except ValueError:
                pass

            MIN_MATCH_COUNT = 10
            if len(good_matches)>MIN_MATCH_COUNT:

                # construct two sets of points - source (the selected object/region points), destination (the current frame points)

                source_pts = np.float32([ keypoints_region[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                destination_pts = np.float32([ keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

                # compute the homography (matrix transform) from one set to the other using RANSAC

                H, mask = cv2.findHomography(source_pts, destination_pts, cv2.RANSAC, 5.0)

                # extract the bounding co-ordinates of the cropped/selected region

                h,w,c = frame.shape
                boundingbox_points = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

                # transform the bounding co-ordinates by homography H

                dst = cv2.perspectiveTransform(boundingbox_points,H)

                # draw the corresponding

                #frame = cv2.polylines(frame,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)

                # draw the matches

                #draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), flags = 0)
                #display_matches = cv2.drawMatches(display_frame,keypoints_region,imgR,keypoints,good_matches,None,**draw_params)
                #cv2.imshow("",display_matches)

        frame_coords = []
        imgR_coords = []

        # getting coords of keypoints
        # code taken from [rayryeng, 2015 - https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python]
        for mat in good_matches:
            frame_index = mat.queryIdx
            imgR_index = mat.trainIdx
            frame_coords.append(keypoints_region[frame_index].pt)
            imgR_coords.append(keypoints[imgR_index].pt)
            
        # get depth of featuure points
        points = project_disparity_to_3d(frame_coords,imgR_coords)

#########################################################################################
        #drawing the boxes and distances on the image - for linear interpolation/finding nearest feature point to midpoint
        
        # linear interpolation of feature points to get estimated depth at every pixel
        if (interpolate):
            
            every_num_pixels = 12 # increase to speed up program but lose accuracy of depth estimation (decrease for opposite)
            interpolated_points = []
            for i in range(len(points)):
                interpolated_points.append(points[i])
                
            image_size = frame.shape
            for x in range(0,image_size[1],every_num_pixels): #width of image
                for y in range(0,image_size[0],every_num_pixels): #height of image

                    #set default values
                    nearest_right = [0,0,0]
                    nearest_left = [0,0,0]
                    dist_right = 1000
                    dist_left = 1000
                    match = False

                    #finding the nearest feature point to the left and right of the pixel
                    
                    for i in range(len(points)):

                        #stop if pixel is already feature point
                        
                        if points[i][0] == x and points[i][1] ==y: 
                            match = True
                            break

                        #calculate the distance between the pixel and the feature point
                        
                        dist = math.sqrt(abs((x**2)-(points[i][0]**2))+abs((y**2)-(points[i][1]**2)))
                        
                        if x<points[i][0]:
                            if dist<dist_left:
                                dist_left = dist
                                nearest_left = points[i]
                        elif x>points[i][0]:
                            if dist<dist_right:
                                dist_right = dist
                                nearest_right = points[i]

                    # adding the new pixel with the linearly interpolated depth to the set of interpolated points
                    if match == False:
                        
                        #stop if no feature points found
                        
                        if nearest_right == [0,0,0] and nearest_left == [0,0,0]:
                            break

                        #take depth of nearest pixel to left if none to right
                        
                        elif nearest_right == [0,0,0]:
                            interpolated_points.append([x,y,nearest_left[2]])

                        #take depth of nearest pixel to right if none to left
                            
                        elif nearest_left == [0,0,0]:
                            interpolated_points.append([x,y,nearest_right[2]])

                        #linear interpolation calculation
                            
                        else:
                            interpolated_points.append([x,y,(nearest_left[2]+((nearest_right[2]-nearest_left[2])*((dist_left-0)/((dist_right+dist_left)-0))))])

            closest_object = 1000
            # draw resulting detections and distances on image and calculate nearest object
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
                 
                for i in range(0,len(interpolated_points)):
                    if interpolated_points[i][0]>70 and interpolated_points[i][1]<image_size[1]-300:
                        if round(interpolated_points[i][0])>round(left+(width/2.5)) and round(interpolated_points[i][0])<round(left+width-(width/2.5)) and round(interpolated_points[i][1])>round(top+(height/4)) and round(interpolated_points[i][1])<round(top+height-(height/3)):
                            new_depth = interpolated_points[i][2]
                            if new_depth<depth:
                                depth = new_depth
                                point = interpolated_points[i]

                # if no feature points in central box
                # (for example if every_num_pixels is high and very few pixel depths are interpolated)
                
                if depth == 1000:

                    # select the midpoint of the object detected
                    
                    middle_x = round(left+(width/2))
                    middle_y = round(top+(height/2))
                    if middle_x<135: # ensure within bounds
                        middle_x = 135

                    # find the nearest interpolated point to the midpoint and take it's depth (less accurate than above)

                    nearest = image_size[0]
                    nearest_points = interpolated_points[0]
                    for i in range(len(interpolated_points)):
                        dist = math.sqrt(abs((middle_x**2)-(interpolated_points[i][0]**2))+abs((middle_y**2)-(interpolated_points[i][1]**2)))
                        if dist<nearest:
                            nearest = dist
                            nearest_point = interpolated_points[i]
                    depth = nearest_point[2]

                    # object only acknowledged is within bounds for object detection
                    # (missing the left, car bonnet and ignoring objects too far away to accurately get
                    # the distance of)
                    
                    if top<300 and left>10 and (width>30 or height>width+10):
                        
                        #updating the depth of the closest object in the frame
                        
                        if depth<closest_object:
                            closest_object = depth
                            
                        # drawing the point at which the depth was taken,
                        # the box around the object and it's distance
                        
                        #cv2.rectangle(display_frame, (int(nearest_point[0]),int(nearest_point[1])), (int(nearest_point[0])+1,int(nearest_point[1])+1),(0,0,255),3)
                        drawPred(display_frame, classes[classIDs[detected_object]], depth, left, top, left + width, top + height, (255, 178, 50))

                else:
                    # object only acknowledged is within bounds for object detection
                    # (missing the left, car bonnet and ignoring objects too far away to accurately get
                    # the distance of)
                    if top<300 and left>10 and (width>30 or height>width+10):
                        
                        # calculate nearest detected scene object
                        
                        if depth<closest_object:
                            closest_object = depth
                            
                        # drawing the point at which the depth was taken,
                        # the box around the object and it's distance
                        
                        #cv2.rectangle(display_frame, (int(point[0]), int(point[1])), (int(point[0])+1, int(point[1])+1), (0,0,255), 3)
                        drawPred(display_frame, classes[classIDs[detected_object]],depth, left, top, left + width, top + height, (255, 178, 50))


        else:

            # for faster, less accurate object range estimation
            # draw resulting detections and distances on image and calculate nearest object
            
            closest_object = 1000
            for detected_object in range(0, len(boxes)):
                box = boxes[detected_object]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                depth = 0

                # control the boundaries of the object and find it's midpoint
                
                if left<0:
                    left =0
                image_size = frame.shape
                if left+width > image_size[1]:
                    width = image_size[1]-left
                middle_x = round(left+(width/2))
                middle_y = round(top+(height/2))
                if middle_x<135: 
                    middle_x = 135

                # find the nearest interpolated point to the midpoint and take it's depth
                
                nearest = image_size[0]
                nearest_points = points[0]
                for i in range(len(points)):
                    dist = math.sqrt(abs((middle_x**2)-(points[i][0]**2))+abs((middle_y**2)-(points[i][1]**2)))
                    if dist<nearest:
                        nearest = dist
                        nearest_point = points[i]
                depth = nearest_point[2]

                # object only acknowledged is within bounds for object detection
                # (missing the left, car bonnet and ignoring objects too far away to accurately get the distance of)

                if top<300 and left>10 and (width>30 or height>width+10):

                    # calculate nearest detected scene object
                    
                    if depth<closest_object:
                        closest_object = depth

                    # drawing the point at which the depth was taken,
                    # the box around the object and it's distance
                    
                    #cv2.rectangle(display_frame, (int(nearest_point[0]),int(nearest_point[1])), (int(nearest_point[0])+1,int(nearest_point[1])+1),(0,0,255),3)
                    drawPred(display_frame, classes[classIDs[detected_object]], depth, left, top, left + width, top + height, (255, 178, 50))

#########################################################################################
        # output image and text, show keys for actions
        
        # print filenames and nearest detected scene object
        print(filename_left)
        if closest_object == 1000:
            closest_object = 0
        print(filename_right,": nearest detected scene object ("+str("{:.1f}".format(closest_object))+"m)")

        #display image

        cv2.imshow('left image',display_frame)
        
        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # use interpolation -i
        # pause - space

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
        elif (key == ord('i')):     # use interpolated points
            interpolate = not(interpolate)

# close all windows

cv2.destroyAllWindows()

#####################################################################

        
        

    
