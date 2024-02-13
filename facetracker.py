import os
import argparse
import traceback
import threading
import queue
import time
from tracker import Tracker
import webcam
import vts
import cv2
cv2.setNumThreads(6)
import maffs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=480)
parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default=0)
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=0.6)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized.", default=3, choices=[0, 1, 2, 3, 4])
parser.add_argument("--preview", type=int, help="Preview the frames sent to the tracker", default=0)
parser.add_argument("--feature-type", type=int, help="Sets which version of feature extraction is used. 0 is my new version that works well for me and allows for some customization, 1 is EmilianaVT's version aka, normal OpenSeeFace operation", default=0, choices=[0, 1])
parser.add_argument("--numpy-threads", type=int, help="Numer of threads Numpy can use, doesn't seem to effect much", default=1)
parser.add_argument("-T","--threads", type=int, help="Numer of threads used for landmark detection. Default is 1 (~15ms per frame on my computer), 2 gets slightly faster frames (~10ms on my computer), more than 2 doesn't seem to help much", default=1)
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

if height > 480:
    target_duration = 1 / (fps - 0.001)
    frameQueueSize = 2
else:
    frameQueueSize = 1
    target_duration = 1 / (fps)

#---Setting up worker threads---

frameQueue = queue.Queue(maxsize=frameQueueSize)
faceQueue = queue.Queue(maxsize=1)
faceInfoQueue = queue.Queue()

#this thread sends requests to Vtube Studio
VTS = vts.VTS()
VTS.targetIP = args.ip
VTS.targetPort = args.port
VTS.silent = silent
VTS.width = args.width
VTS.height = args.height
VTS.faceInfoQueue = faceInfoQueue

packetSenderThread = threading.Thread(target = VTS.start)
packetSenderThread.daemon = True
packetSenderThread.start()

#this thread gets images from the webcam
Webcam = webcam.Webcam()
Webcam.width = args.width
Webcam.height = args.height
Webcam.fps = args.fps
Webcam.mirror = args.mirror_input
Webcam.targetBrightness = min(max(args.target_brightness, 0.25), 0.75)
Webcam.preview = (args.preview == 1)
Webcam.frameQueue = frameQueue
Webcam.faceQueue = faceQueue

webcamThread = threading.Thread(target=Webcam.start)
webcamThread.daemon = True
webcamThread.start()


webcamStats = maffs.Stats()
trackingTimeStats = maffs.Stats()
frameTimeStats = maffs.Stats()
latencyStats = maffs.Stats()

lateFrames = 0
frame_count = 0
frame_start = 0.0
sleepTime = 0.0

tracker = Tracker(args)

#don't start until the webcam is ready, then give it a little more time to fill it's buffer
frameQueue.get()
time.sleep(target_duration-0.01)

trackingStart = time.perf_counter()
try:
    while True:
        frame_start = time.perf_counter()
        frame_count += 1
        frame = frameQueue.get()
        frame_get = time.perf_counter()
        webcamStats.update(frame.cameraLatency)

        faceInfo, frame.face  = tracker.predict(frame)

        if frame.face is not None:
            if faceQueue.qsize() < 1:
                faceQueue.put(frame)

        frameTime = time.perf_counter() - frame_get
        trackingTimeStats.update(frameTime)

        if visualizeFlag:
            visualize(frame, faceInfo)

        duration = (time.perf_counter() - frame_start)

        sleepTime = target_duration - duration
        if sleepTime > 0:
            time.sleep(sleepTime)
        else:
            print("Cannot maintain framerate")
            lateFrames += 1

        #If we don't have something to send to Vtube Studio we don't
        if faceInfo is not None:
            faceInfoQueue.put(faceInfo)
        else:
            print("No data sent to VTS")

        timeSinceLastFrame = (time.perf_counter() - frame_start)
        frameTimeStats.update(timeSinceLastFrame)
        latency = time.perf_counter() - frame.startTime
        latencyStats.update(latency)

except KeyboardInterrupt:
    if not silent:
        print("Quitting")

#printing statistics on close
#it makes identifying problems easier

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
print(f"Frames: {frame_count}")
print(f"Late frames: {lateFrames}, {((lateFrames/frame_count)*100):.2f}% of frames")

os._exit(0)
