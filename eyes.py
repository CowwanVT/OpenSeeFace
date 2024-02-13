import numpy as np
import math
import cv2
cv2.setNumThreads(6)
import maffs

def rotate_image(image, a, center): #twice per frame, 0.2ms - 0.25ms each, improved to 0.15ms - 0.25ms
    center = (float(center[0]), float(center[1]))
    a = math.degrees(a)
    M = cv2.getRotationMatrix2D(center,a, 1.0)
    (h, w) = image.shape[:2]
    image = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_CUBIC)
    return image

class Eye():
    mean = np.float32(np.array([-2.1179, -2.0357, -1.8044]))
    std = np.float32(np.array([0.0171, 0.0175, 0.0174]))
    def __init__(self,index):
        self.index = index
        self.state = [1.,0.,0.,0.]
        self.image = None
        self.info = None
        self.results = None
        self.confidence = 0
        self.lastEyeState = [0.,0.]
        self.innerPoint = None
        self.outerPoint = None
        self.stdDev = maffs.Stats()
        self.xStats = maffs.Stats()
        self.yStats = maffs.Stats()

    def prepare_eye(self, faceFrame):
        self.state = [1.,0.,0.,0.]
        im = faceFrame
        (x1, y1), (x2, y2), a = self.corners_to_eye(im.shape)

        #rotating an image is expensive and reduces clarity
        #so I just don't if it's a relatively small angle
        if math.degrees(a) > 7.5 and math.degrees(a) < 352.5:
            im = rotate_image(im, a, self.outerPoint)
        im = im[int(y1):int(y2), int(x1):int(x2)]

        if np.prod(im.shape) < 1:
            self.image = None
            self.info = None
            return

        if self.index == 1:
            im = cv2.flip(im, 1)
        scale = [(x2 - x1)/ 32., (y2 - y1)/ 32.]

        im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_CUBIC)
        im = im.astype(np.float32) * self.std + self.mean
        im = np.expand_dims(im, 0)
        self.image = np.transpose(im, (0,3,2,1))
        self.info = [x1, y1, scale, self.outerPoint, a]
        return

    def corners_to_eye(self, shape):

        h, w, _ = shape
        c1 = np.array(self.outerPoint)
        c2 = np.array(self.innerPoint)

        a = math.atan2(*(c2 - c1)[::-1])
        a = a % (2*math.pi)
        c2 = maffs.rotate(c1, c2, a)
        center = (c1 + c2) / 2.0

        radius = np.linalg.norm(c1 - c2)
        radius = [radius * 0.7, radius * 0.6]
        upper_left = maffs.clamp_to_im(center - radius, w, h)
        lower_right = maffs.clamp_to_im(center + radius, w, h)
        return upper_left, lower_right, a

    def calculateEye(self):

        e_x, e_y, scale, reference, angles = self.info
        confidenceThreshold = self.stdDev.getMean() - 2 * self.stdDev.getSampleVariance()

        m = self.results[0].argmax()
        x = m // 8
        y = m % 8

        p=self.results[1][x, y]
        p = maffs.clamp(p, 0.00001, 0.9999)
        off_x = math.log(p/(1-p))
        eye_x = 4.0 *(x + off_x)

        p=self.results[2][x, y]
        p = maffs.clamp(p, 0.00001, 0.9999)
        off_y = math.log(p/(1-p))
        eye_y = 4.0 * (y + off_y)

        if eye_x < self.lastEyeState[1]:
            delta = self.lastEyeState[1] - eye_x
            delta = self.xStats.clamp(delta)
            eye_x = self.lastEyeState[1] - delta
        if eye_x < self.lastEyeState[1]:
            delta = eye_x - self.lastEyeState[1]
            delta = self.xStats.clamp(delta)
            eye_x = self.lastEyeState[1] + delta

        if eye_y < self.lastEyeState[0]:
            delta = self.lastEyeState[0] - eye_y
            delta = self.yStats.clamp(delta)
            eye_y = self.lastEyeState[0] - delta
        if eye_y < self.lastEyeState[0]:
            delta = eye_y - self.lastEyeState[0]
            delta = self.yStats.clamp(delta)
            eye_y = self.lastEyeState[0] + delta



        #if eye movements are below 3 standard deviations of the average the movement rejected
        if self.results[0][x,y] < confidenceThreshold:
            eye_y = self.lastEyeState[0]
            eye_x = self.lastEyeState[1]

        self.lastEyeState = [eye_y, eye_x]

        if self.index == 1:
            eye_x = (32. - eye_x)
        eye_x = e_x + scale[0] * eye_x
        eye_y = e_y + scale[1] * eye_y

        eye_x, eye_y = maffs.rotate(reference, (eye_x, eye_y), -angles)
        eye_x, eye_y = (eye_x, eye_y) + self.offset

        self.confidence = self.results[0][x,y]
        self.stdDev.update(self.confidence)
        self.state  = [1.0, eye_y, eye_x, self.confidence]
        return

class EyeTracker():
    np.float32(np.array([-2.1179, -2.0357, -1.8044]))
    std = np.float32(np.array([0.0171, 0.0175, 0.0174]))
    def __init__(self):
        self.leftEye = Eye(1)
        self.rightEye = Eye(0)
        self.offset = None

    def get_eye_state(self, models, frame, lms):

        lms, faceFrame = self.extract_face(frame, np.array(lms)[:,0:2][:,::-1])

        self.rightEye.offset = self.offset
        self.leftEye.offset = self.offset
        self.rightEye.innerPoint = lms[39,0:2]
        self.rightEye.outerPoint = lms[36,0:2]
        self.leftEye.innerPoint = lms[45,0:2]
        self.leftEye.outerPoint = lms[42,0:2]

        self.rightEye.prepare_eye(faceFrame)
        self.leftEye.prepare_eye(faceFrame)

        if self.rightEye.image is None or self.leftEye.image is None:
            return [[1.,0.,0.,0.],[1.,0.,0.,0.]]    #Early exit if one of the eyes doesn't have data
        both_eyes = np.concatenate((self.rightEye.image, self.leftEye.image))

        self.rightEye.results, self.leftEye.results = models.gazeTracker.run([], {"input": both_eyes})[0]

        self.rightEye.calculateEye()
        self.leftEye.calculateEye()

        return [self.rightEye.state, self.leftEye.state]

    def extract_face(self, frame, lms):
        x1, y1 = lms.min(0)
        x2, y2 = lms.max(0)

        x1 = frame.clampX(x1)
        y1 = frame.clampY(y1)

        self.offset = np.array((x1, y1))
        lms = (lms[:, 0:2] - self.offset).astype(int)
        image = frame.crop(x1, x2, y1, y2)

        return lms, image
