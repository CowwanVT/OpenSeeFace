import os
import numpy as np
import math
import cv2
import onnxruntime
import time



def clamp_to_im(pt, w, h): #8 times per frame, but that only accounts for 0.005ms
    x=max(pt[0],0)
    y=max(pt[1],0)
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    return (int(x),int(y+1))

def rotate(origin, point, a): #runs 22 times per frame
    x, y = point - origin
    cosa = math.cos(-a)
    sina = math.sin(-a)
    qx = origin[0] + cosa * x - sina * y
    qy = origin[1] + sina * x + cosa * y
    return qx, qy

def clamp (value, minimum, maxium):
    return max(min(value,maxium),minimum)

def rotate_image(image, a, center): #twice per frame, 0.2ms - 0.25ms each, improved to 0.15ms - 0.25ms
    M = cv2.getRotationMatrix2D(center, math.degrees(a), 1.0)
    (h, w) = image.shape[:2]
    image = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_CUBIC)
    return image

class EyeTracker():
    def __init__(self):
        self.last_eye_state = [[0.,0.],[0.,0.]]
        self.average_eye_confidence = [1.0,1.0]
        self.average_eye_confidence_variance = [0.0,0.0]
        self.eye_std_dev = [0.0,0.0]
        self.eye_tracking_frames = 0
        self.mean = np.float32(np.array([-2.1179, -2.0357, -1.8044]))
        self.std = np.float32(np.array([0.0171, 0.0175, 0.0174]))

        model_base_path = os.path.join(os.path.dirname(__file__), os.path.join("models"))
        providersList = onnxruntime.capi._pybind_state.get_available_providers()
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.gaze_model = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_gaze32_split_opt.onnx"), sess_options=options, providers=providersList)


    def get_eye_state(self,face_frame, lms, offset):
        eye_state = [[1.,0.,0.,0.],[1.,0.,0.,0.]]   #Array constructed early so it can be used for an early exit
        self.eye_tracking_frames+=1

        #I've tried to thread this because they don't need to happen in sequence
        #but it doesn't do any good because python threads kinda suck
        #I'd need a persistent right eye and left eye process and that feels excessive
        right_eye, REyeInfo = self.prepare_eye(face_frame, lms[36,0:2], lms[39,0:2], False)
        left_eye, LEyeInfo = self.prepare_eye(face_frame,lms[42,0:2],  lms[45,0:2], True)

        if right_eye is None or left_eye is None:
            return eye_state    #Early exit if one of the eyes doesn't have data
        both_eyes = np.concatenate((right_eye, left_eye))

        results = self.gaze_model.run([], {"input": both_eyes})[0]
        eye_state[0] = self.calculateEye(results[0], REyeInfo, offset,0)
        eye_state[1] = self.calculateEye(results[1], LEyeInfo, offset,1)
        return eye_state

    def calculateStandardDeviation(self, conf, eye):

        avgRatio = 1/self.eye_tracking_frames
        self.average_eye_confidence[eye] = (self.average_eye_confidence[eye]*(1-avgRatio))+ (conf * avgRatio)

        avgRatio = 1/(self.eye_tracking_frames + 1)
        self.average_eye_confidence_variance[eye] = pow(((conf - self.average_eye_confidence[eye]) * avgRatio) + (self.average_eye_confidence_variance[eye] * (1-avgRatio)), 2)

        self.eye_std_dev[eye] = math.sqrt(self.average_eye_confidence_variance[eye])

    def calculateEye(self, results, eye_info , offset, eye):

        e_x, e_y, scale, reference, angles = eye_info
        confidenceThreshold = self.average_eye_confidence[eye] - 2 * self.eye_std_dev[eye]

        m = results[0].argmax()
        x = m // 8
        y = m % 8

        p=results[1][x, y]
        p = clamp(p, 0.00001, 0.9999)
        off_x = math.log(p/(1-p))
        eye_x = 4.0 *(x + off_x)

        p=results[2][x, y]
        p = clamp(p, 0.00001, 0.9999)
        off_y = math.log(p/(1-p))
        eye_y = 4.0 * (y + off_y)

        #I'm proud of this
        #it's my "anti-Cali" functionality
        #if eye movements are below 2 standard deviations of the average the movement is severely limited
        #the eyes are handled independenly because my testing showed that one eye tended to have a higher condifence than the other
        limit = max((results[0][x,y]- 0.5)*(2/confidenceThreshold),0)
        if results[0][x,y] < confidenceThreshold:
            Delta = eye_y - self.last_eye_state[eye][0]
            eye_y = self.last_eye_state[eye][0] + clamp(Delta, -limit, limit)
            Delta = eye_x - self.last_eye_state[eye][1]
            eye_x = self.last_eye_state[eye][1] + clamp(Delta, -limit, limit)

        self.last_eye_state[eye] = [eye_y, eye_x]

        if eye == 1:
            eye_x = (32. - eye_x)
        eye_x = e_x + scale[0] * eye_x
        eye_y = e_y + scale[1] * eye_y

        eye_x, eye_y = rotate(reference, (eye_x, eye_y), -angles)
        eye_x, eye_y = (eye_x, eye_y) + offset
        self.calculateStandardDeviation(results[0][x,y], eye)
        return [1.0, eye_y, eye_x, results[0][x,y]]

    def prepare_eye(self,im, outer_pt, inner_pt, flip):
        (x1, y1), (x2, y2), center,  reference, a = self.corners_to_eye((outer_pt, inner_pt), im.shape[1], im.shape[0])
        #rotating an image is expensive and reduces clarity
        #so I just don't if it's a relatively small angle
        if math.degrees(a) > 7.5 and math.degrees(a) < 352.5:
            im = rotate_image(im, a, reference)

        im = im[int(y1):int(y2), int(x1):int(x2)]
        if np.prod(im.shape) < 1:
            return None, None, None, None, None, None
        if flip:
            im = cv2.flip(im, 1)
        scale = [(x2 - x1)/ 32., (y2 - y1)/ 32.]
        im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_CUBIC)
        im = im.astype(np.float32) * self.std + self.mean
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,2,1))
        return [im, [x1, y1, scale, reference, a]]

    def corners_to_eye(self,  corners, w, h):
        c1, c2 = np.array(corners, dtype=float)
        a = math.atan2(*(c2 - c1)[::-1]) % (2*math.pi)
        c2 = rotate(c1, c2, a)
        center = (c1 + c2) / 2.0
        radius = np.linalg.norm(c1 - c2)
        radius = [radius * 0.7, radius * 0.6]
        upper_left = clamp_to_im(center - radius, w, h)
        lower_right = clamp_to_im(center + radius, w, h)
        return upper_left, lower_right, center,  c1, a