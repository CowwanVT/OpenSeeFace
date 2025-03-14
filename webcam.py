import os
os.environ["OMP_NUM_THREADS"] = str(1)
import dshowcapture
import cv2
cv2.setNumThreads(6)
import time
import numpy as np
import math


class Webcam():
    def __init__(self):
        self.cameraID = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.preview = False
        self.mirror = None
        self.frameQueue = None
        self.targetFrametime = 0
        self.targetBrightness = 0.55
        self.bufferFrames = -1
        self.maxQueueSize = 1

    def initialize(self):
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(int(self.cameraID), cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(int(self.cameraID), cv2.CAP_V4L2   )
        if self.height > 480:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if self.bufferFrames >= 0:
            self.cap.set(38, self.bufferFrames)


        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.read()
        time.sleep(3/self.fps)
        self.targetFrametime = 1/self.fps

    def start(self):
        self.initialize()
        while self.cap.isOpened():
            frameStart= time.perf_counter()
            frame = self.getFrame()
            if frame.ret:
                if self.frameQueue.qsize() < self.maxQueueSize :
                    self.frameQueue.put(frame)
                if self.preview:
                    cv2.imshow("test",cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            frameDuration = time.perf_counter() - frameStart
            sleepTime = self.targetFrametime - frameDuration
            if sleepTime > 0:
                time.sleep(sleepTime)
            else:
                time.sleep(self.targetFrametime/10)


    def getFrame(self):
        cameraStart = time.perf_counter()

        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mirror and ret:
            image = cv2.flip(self.image, 1)
        frame = Frame(ret, image, cameraStart, self.targetBrightness )
        return frame

class Frame():
    def __init__(self, ret, image, cameraStart, targetBrightness):
        self.face = None
        self.ret = ret
        self.cameraLatency = time.perf_counter() - cameraStart
        self.image = image
        self.adjustedImage = None
        if ret:
            self.startTime = time.perf_counter()
            self.width = self.image.shape[1]
            self.height = self.image.shape[0]
        self.targetBrightness = targetBrightness

    def applyGamma(self):

        y1, y2, x1, x2 = self.face
        img_yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
        gamma = self.calculateGamma(img_yuv[y1:y2,x1:x2,0])

        #building a lookup table on the fly
        lookupTable = np.array(range(256))
        loopupTable = lookupTable/255
        lookupTable = np.power(loopupTable, gamma)*255
        img_yuv[:,:,0] = lookupTable[img_yuv[:,:,0]]
        self.adjustedImage = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    def calculateGamma(self, greyscaleImage):

        averageBrightness = np.mean(greyscaleImage)/256
        gamma = math.log(self.targetBrightness, averageBrightness)

        if math.isnan(gamma):
            gamma = 0.7

        gamma = max(gamma, 0.5)
        gamma = min(gamma, 1.5)
        return gamma

    def crop(self, x1, x2, y1, y2):

        x1 = self.clampX(x1)
        x2 = self.clampX(x2)
        y1 = self.clampY(y1)
        y2 = self.clampY(y2)

        if self.adjustedImage is not None:
            crop = self.adjustedImage[y1:y2,x1:x2]
        else:
            crop = self.image[y1:y2,x1:x2]
        return crop

    def cropFace(self, x1, x2, y1, y2):
        x1 = self.clampX(x1)
        x2 = self.clampX(x2)
        y1 = self.clampY(y1)
        y2 = self.clampY(y2)
        self.face = [y1, y2, x1, x2]
        self.applyGamma()

        return self.adjustedImage[y1:y2,x1:x2]

    def clampX(self, x):
        x = max(x, 0)
        x = min (x, self.width - 1)
        return int(x)

    def clampY(self, y):
        y = max(y, 0)
        y = min (y, self.width - 1)
        return int(y)
