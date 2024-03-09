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
        self.width = 0
        self.height = 0
        self.fps = 0
        self.targetFrameTime = 0.0
        self.gamma = 0.7
        self.preview = False
        self.mirror = None
        self.frameQueue = None
        self.faceQueue = None
        self.targetBrightness = 0.55

    def initialize(self):
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)

        if self.height > 480:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(38, 2)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.targetFrameTime = 1 / (self.fps)
        time.sleep(0.1)
        self.cap.read()
        time.sleep(0.1)

    def start(self):
        self.initialize()
        while self.cap.isOpened():
            frameStart= time.perf_counter()
            frame = self.getFrame()
            if frame.ret:
                self.frameQueue.put(frame)
                if self.preview:
                    cv2.imshow("test",cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            if self.faceQueue.qsize() > 0:
                self.updateGamma()
            sleepTime = self.targetFrameTime - (time.perf_counter() - frameStart)
            sleepTime = max(sleepTime, 0)
            time.sleep(sleepTime)

    def getFrame(self):
        self.cap.set(cv2.CAP_PROP_GAIN, 0)

        cameraStart = time.perf_counter()

        ret, image = self.cap.read()
        if self.mirror and ret:
            image = cv2.flip(self.image, 1)
        frame = Frame(ret, image, self.gamma, cameraStart )
        return frame

    def updateGamma(self):
        frame = self.faceQueue.get()

        crop = frame.cropGreyscale()
        averageBrightness = np.mean(crop)/256
        gamma = math.log(self.targetBrightness, averageBrightness)
        if math.isnan(gamma):
            gamma = 0.7
        if gamma < 0.5:
            gamma = 0.5
        if gamma > 1.5:
            gamma = 1.5
        self.gamma = gamma
        return

class Frame():
    def __init__(self, ret, image, gamma, cameraStart):
        self.greyscale = None
        self.face = None
        self.ret = ret
        self.cameraLatency = time.perf_counter() - cameraStart
        if ret:
            self.image = self.applyGamma(image, gamma)
            self.startTime = time.perf_counter()
            self.width = self.image.shape[1]
            self.height = self.image.shape[0]

    def applyGamma(self,image, gamma):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        #saving a copy of the brightness channel so I can use it to adjust the gamma based on where the face is in that frame
        self.greyscale = img_yuv[:,:,0].copy()
        #building a lookup table on the fly
        lookupTable = np.array(range(256))
        loopupTable = lookupTable/255
        lookupTable = np.power(loopupTable, gamma)*255
        img_yuv[:,:,0] = lookupTable[img_yuv[:,:,0]]
        #I convert the image to RBG here because it was getting repeatedly converted in the face tracking
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return image

    def crop(self, x1, x2, y1, y2):
        x1 = self.clampX(x1)
        x2 = self.clampX(x2)
        y1 = self.clampY(y1)
        y2 = self.clampY(y2)
        crop = self.image[y1:y2,x1:x2]
        return crop

    def cropGreyscale(self):
        x,y,w,h = self.face
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        x1 = self.clampX(x1)
        x2 = self.clampX(x2)
        y1 = self.clampY(y1)
        y2 = self.clampY(y2)
        crop = self.greyscale[y1:y2,x1:x2]
        return crop


    def clampX(self, x):
        x = max(x, 0)
        x = min (x, self.width - 1)
        return int(x)

    def clampY(self, y):
        y = max(y, 0)
        y = min (y, self.width - 1)
        return int(y)

