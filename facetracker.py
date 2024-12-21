import os
import argparse
import traceback
import threading
import queue
import time
from tracker import Tracker
import webcam
import cv2
cv2.setNumThreads(6)
import maffs
import api

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-a", "--api-port", type=int, help="Set port for Vtube Studio API", default=8001)
parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=480)
parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default=0)
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=0.7)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.7)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized.", default=3, choices=[0, 1, 2, 3, 4])
parser.add_argument("--preview", type=int, help="Preview the frames sent to the tracker", default=0)
parser.add_argument("--numpy-threads", type=int, help="Numer of threads Numpy can use, doesn't seem to effect much", default=1)
parser.add_argument("-T","--threads", type=int, help="Numer of threads used for landmark detection. Default is 2", default=4)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking", default=0)
parser.add_argument("--target-brightness", type=float, help="range 0.25-0.75, Target brightness of the brightness adjustment. Defaults to 0.55", default = 0.55)

args = parser.parse_args()

def visualize(frame, face):

    y1, x1, _ = face.lms[0:66].min(0)
    y2, x2, _ = face.lms[0:66].max(0)
    x1 -= 0.25*(x2 - x1)
    x2 += 0.25*(x2 - x1)
    y1 -= 0.5*(y2 - y1)
    y2 += 0.1*(y2 - y1)
    h = y2 - y1
    ratio = 720 / h
    h = int(ratio * frame.height)
    w = int(ratio * frame.width)

    image = cv2.resize(frame.image, (w,h), interpolation=cv2.INTER_CUBIC)

    for pt_num, (x,y,c) in enumerate(face.lms):
        x = int(x * ratio + 0.5)
        y = int(y * ratio + 0.5)
        image = cv2.putText(image, str(pt_num), (int(y+5), int(x-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0))
        color = (255, 0, 0)
        if not (x < 0 or y < 0 or x >= h or y >= w):
            cv2.circle(image, (y, x), 3, color, -1)
    x1 *= ratio
    x2 *= ratio
    y1 *= ratio
    y2 *= ratio

    x1 = int(max(x1, 0))
    y1 = int(max(y1, 0))
    x2 = int(min(x2, w - 1))
    y2 = int(min(y2, h - 1))

    image = image[y1:y2,x1:x2]

    cv2.imshow("Visualization",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

#processing args
height = args.height
width = args.width
visualizeFlag = (args.visualize == 1)
silent = (args.silent == 1)
fps = args.fps

#---Setting up worker threads---

frameQueue = queue.Queue()
featureQueue = queue.Queue()

#Sending API calls on a separate thread so they don't block anything else
api = api.VtubeStudioAPI()
api.ip = args.ip
api.port = args.api_port
api.featureQueue = featureQueue
apiThread = threading.Thread(target = api.start)
apiThread.daemon = True
apiThread.start()

#this thread gets images from the webcam
Webcam = webcam.Webcam()
Webcam.cameraID = args.capture
Webcam.width = args.width
Webcam.height = args.height
Webcam.fps = args.fps
Webcam.mirror = args.mirror_input
Webcam.targetBrightness = min(max(args.target_brightness, 0.25), 0.75)
Webcam.preview = (args.preview == 1)
Webcam.frameQueue = frameQueue

webcamThread = threading.Thread(target=Webcam.start)
webcamThread.daemon = True
webcamThread.start()

#---Setting up statistic trackers---
webcamStats = maffs.Stats()
trackingTimeStats = maffs.Stats()
frameTimeStats = maffs.Stats()
latencyStats = maffs.Stats()

frameCount = 0
frameStart = 0.0
sleepTime = 0.0
tracker = Tracker(args)

#don't start until the webcam is ready, then give it a little more time to fill it's buffer
frameQueue.get()
time.sleep(1 / fps)

trackingStart = time.perf_counter()

frame_get = time.perf_counter()
frame = None
#---The actual main loop---
try:
    faceInfo = None
    while True:
        frameStart = time.perf_counter()
        frameCount += 1

        frame = frameQueue.get()
        frame_get = time.perf_counter()
        webcamStats.update(frame.cameraLatency)

        faceInfo = tracker.predict(frame)

        if visualizeFlag:
            visualize(frame, faceInfo)

        frameTime = time.perf_counter() - frame_get
        trackingTimeStats.update(frameTime)

        #While timing is based on the webcam thread, I use this to even out the frame timing
        targetTrackingTime = trackingTimeStats.getMean() + (3*trackingTimeStats.getVariance())
        sleepTime = targetTrackingTime - frameTime
        if sleepTime > 0:
            time.sleep(sleepTime)

        if faceInfo is not None:
            if featureQueue.qsize() < 1:
                featureQueue.put(faceInfo.currentAPIFeatures)
                latency = time.perf_counter() - frame.startTime
                latencyStats.update(latency)

        else:
            print("No data sent to VTS")

        duration = (time.perf_counter() - frameStart)
        timeSinceLastFrame = (time.perf_counter() - frameStart)
        frameTimeStats.update(timeSinceLastFrame)


except KeyboardInterrupt:
    if not silent:
        print("Quitting")

#---Printing some potentially useful stats---

#time from getting the frame from the webcam to sending face data to VTS
print(f"Peak latency: {(latencyStats.maximum * 1000):.3f}ms")
print(f"Average latency: {(latencyStats.getMean() * 1000):.3f}ms")

#time taken to get frame from webcam
print(f"Peak camera latency: {(webcamStats.maximum * 1000):.3f}ms")
print(f"Average camera latency: {(webcamStats.getMean() * 1000):.3f}ms")

#time between packets sent to VTS
print(f"Peak time between frames: {(frameTimeStats.maximum * 1000):.3f}ms")
print(f"Average time between frames: {(frameTimeStats.getMean() * 1000):.3f}ms")

#how long face detection took
print(f"Peak tracking time: {(trackingTimeStats.maximum * 1000):.3f}ms")
print(f"Average tracking time: { (trackingTimeStats.getMean() * 1000):.3f}ms")

#how long the app ran
print(f"Run time (seconds): {(time.perf_counter() - trackingStart):.2f} s")
print(f"Frames: {frameCount}")

os._exit(0)
