#!/usr/bin/env python3

"""
The code is the same as for Tiny Yolo V3 and V4, the only difference is the blob file
- Tiny YOLOv3: https://github.com/david8862/keras-YOLOv3-model-set
- Tiny YOLOv4: https://github.com/TNTWEN/OpenVINO-YOLOV4
"""

# https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/
# https://docs.luxonis.com/projects/api/en/latest/samples/SpatialDetection/spatial_tiny_yolo/


from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import serial  # Potential issue: https://stackoverflow.com/questions/11403932/python-attributeerror-module-object-has-no-attribute-serial
import pynmea2

# Get argument first
nnPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
if 1 < len(sys.argv):
    arg = sys.argv[1]
    if arg == "yolo3":
        nnPath = str((Path(__file__).parent / Path('../models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    elif arg == "yolo4":
        nnPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    else:
        nnPath = arg
else:
    print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# tiny yolo v4 label texts
labelMap = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
        ]

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(20)
detectionNetwork.setCoordinateSize(4)
# detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detectionNetwork.setAnchors([10.0,
                13.0,
                16.0,
                30.0,
                33.0,
                23.0,
                30.0,
                61.0,
                62.0,
                45.0,
                59.0,
                119.0,
                116.0,
                90.0,
                156.0,
                198.0,
                373.0,
                326.0])

detectionNetwork.setAnchorMasks({ "side80": [
                    0,
                    1,
                    2
                ],
                "side40": [
                    3,
                    4,
                    5
                ],
                "side20": [
                    6,
                    7,
                    8
                ] })

# detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline, usb2Mode=True) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.imshow(name, frame)
            # GPS, is this a multi threaded program?
            # if the detected object is human
            # Show the frame
            # don't need to show UI
            # add calibration
            # markdown to quickly document flow
            if (labelMap[detection.label] == "person"):
                ts = str(int(time.time()))
                f = open("/home/yuxiang/code/2023/trash/customizedYoloFile.txt", "a")
                content = "human detected at " 
                content = content + ts 
                content = content + " " 
                gpsCoordinates = getGPScoordinates()
                # GPS not synchronized, cache GPS while lower the detection rate,  15 FPS is not required
                # Null and None are not the same
                # gpsCoordinates != "" or   
                # if gpsCoordinates is not None :  
                if gpsCoordinates != "":
                    content = content + gpsCoordinates + "\n"
                    f.write(content)
                    f.close()
                    fileName = '/home/yuxiang/code/2023/trash/customizedYolo'+ ts + '.jpg'
                    cv2.imwrite(fileName, frame)



            
    def getGPScoordinates() -> str:
        # Todo: Try and catch
        # Todo: Debug + Logging
        # Todo: object lifetime, do they get deleted?
        # Todo: feed coordinates to a NodeRed map or a local map?
        # Todo: optional GPS, currently assumes GPS is always there?
        # Thoughts: store in a txt file, file contains 
        # the name of the picture with the GPS coordinates
        # put focus back to training garbage net? or crack nets
        # When do we call this project is good enough for a prototype?
        port="/dev/ttyACM0"
        ser=serial.Serial(port, baudrate=9600, timeout=0.5)
        # Delete # dataout = pynmea2.NMEAStreamReader()
        # converting bytes to string: https://stackoverflow.com/questions/606191/convert-bytes-to-a-string
        newdata=ser.readline().decode("utf-8")
        if newdata[0:6] == '$GNRMC':
            newmsg=pynmea2.parse(newdata)
            lat=newmsg.latitude
            lng=newmsg.longitude
            # gps =  " \"name\":\"map\", \"lat\":" + lat + ", \"lon\":" + lng +" "
            gps = "Latitude=" + str(lat) + " Longitude=" + str(lng)
            # Todo: upgrade return to a Json Object
            return gps
        return ""

    while True:
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break