import cv2
import argparse
import numpy as np
import time

import sys
import argparse
import pathlib
sys.path.insert(1,str(pathlib.Path.cwd().parents[0])+"/common")

import cv2
import utils as util

# Function to detect cars in a given frame using YOLOv5 model
def detect_cars(frame, net, ln, confidence_thresh=0.5, overlap_thresh=0.3):
    (H, W) = frame.shape[:2]
    results = []

    # Preprocess the frame and pass it through the network to obtain the detections
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Loop over the detections and filter out the weak ones
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_thresh and classID == 2: # 2 represents car class
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Add the detection to the list of results
                results.append((x, y, int(width), int(height)))

    # Apply non-maxima suppression to remove overlapping detections
    boxes = np.array([r for r in results])
    confidences = [1.0 for _ in range(len(results))]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, overlap_thresh)

    # Return the filtered detections
    detections = []
    for i in indices:
        i = i[0]
        (x, y, w, h) = boxes[i]
        detections.append((x, y, x + w, y + h))

    return detections

# Function to calculate the waiting time for a lane based on the number of cars detected
def calculate_wait_time(num_cars):
    if num_cars == 0:
        return 0
    elif num_cars == 1:
        return 5
    elif num_cars == 2:
        return 10
    else:
        return 15


# Main function
def main(sources):
    # Load the YOLOv5 model
    net = cv2.dnn.readNet(str(pathlib.Path.cwd().parents[0]) + "/models/yolov5s.onnx")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ln = net.getUnconnectedOutLayersNames()  # returns the name of output layer
    model_path = "models/yolov5s.onnx"
    net = cv2.dnn.readNet(model_path)
    ln = net.getUnconnectedOutLayersNames()

    # Initialize the video feeds for each lane
    cap1 = cv2.VideoCapture(sources[0])
    cap2 = cv2.VideoCapture(sources[1])
    cap3 = cv2.VideoCapture(sources[2])
    cap4 = cv2.VideoCapture(sources[3])

    # Initialize the lanes and their waiting times
    lanes = {
        "lane1": {"frame": None, "wait_time": 0},
        "lane2": {"frame": None, "wait_time": 0},
        "lane3": {"frame": None, "wait_time": 0},
        "lane4": {"frame": None, "wait_time": 0},
    }

    # Loop over the frames from the video feeds
    while True:

        while True:
            # Read the frames from each lane's video feed
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()
            ret4, frame4 = cap4.read()

            # Process the frames only if they were read successfully
            if ret1:
                detections1 = detect_cars(frame1, net, ln)
                lanes["lane1"]["frame"] = frame1
                lanes["lane1"]["wait_time"] = calculate_wait_time(len(detections1))

            if ret2:
                detections2 = detect_cars(frame2, net, ln)
                lanes["lane2"]["frame"] = frame2
                lanes["lane2"]["wait_time"] = calculate_wait_time(len(detections2))

            if ret3:
                detections3 = detect_cars(frame3, net, ln)
                lanes["lane3"]["frame"] = frame3
                lanes["lane3"]["wait_time"] = calculate_wait_time(len(detections3))

            if ret4:
                detections4 = detect_cars(frame4, net, ln)
                lanes["lane4"]["frame"] = frame4
                lanes["lane4"]["wait_time"] = calculate_wait_time(len(detections4))

            # Display the frames and the wait times
            util.display_frames(lanes)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determines duaration based on car count on images")
    parser.add_argument("--sources", help="video feeds to be infered on, the videos must reside in the datas folder",
                        type=str, default="video1.mp4,video5.mp4,video2.mp4,video3.mp4")
    args = parser.parse_args()

    sources = args.sources
    sources = sources.split(",")
    print(type(sources))
    main(sources)

