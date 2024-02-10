import os
import numpy as np
import onnxruntime
import cv2
cv2.setNumThreads(6)
import time
import facedetection
import eyes
import landmarks
import face


def clamp_to_im(pt, w, h):
    x = pt[0]
    y = pt[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    return (int(x), int(y+1))

class Models():

    models = [
        "lm_model0_opt.onnx",
        "lm_model1_opt.onnx",
        "lm_model2_opt.onnx",
        "lm_model3_opt.onnx",
        "lm_model4_opt.onnx"]

    def __init__(self, threads,  model_type=3):
        self.model_type = model_type

        model = self.models[self.model_type]
        model_base_path = os.path.join(os.path.dirname(__file__), os.path.join("models"))
        providersList = onnxruntime.capi._pybind_state.get_available_providers()
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self.landmarks = onnxruntime.InferenceSession(os.path.join(model_base_path, model), sess_options=options, providers=providersList)

        self.faceDetection = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_detection_opt.onnx"), sess_options=options, providers=providersList)

        self.gazeTracker = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_gaze32_split_opt.onnx"), sess_options=options, providers=providersList)

class Tracker():
    targetimageSize = [224, 224]
    std = np.float32(np.array([0.0171, 0.0175, 0.0174]))
    mean = np.float32(np.array([-2.1179, -2.0357, -1.8044]))
    def __init__(self, width, height, featureType, threads, model_type=3, detection_threshold=0.6, threshold=0.6, silent=False):

        self.detection_threshold = detection_threshold
        self.EyeTracker = eyes.EyeTracker()
        self.model = Models(threads, model_type = model_type)

        # Image normalization constants
        self.width = width
        self.height = height

        self.threshold = threshold
        self.face = None
        self.face_info = face.FaceInfo(featureType)

    #scales cropped frame before sending to landmarks
    def preprocess(self, im):
        im = cv2.resize(im ,self.targetimageSize , interpolation=cv2.INTER_CUBIC) * self.std + self.mean
        im = np.expand_dims(im, 0)
        return np.transpose(im, (0,3,1,2))

    def cropFace(self, frame):
        duration_pp = 0.0
        x,y,w,h = self.face

        crop_x1 = x - (w * 0.1)
        crop_y1 = y - (h * 0.125)
        crop_x2 = x + w + (w * 0.1)
        crop_y2 = y + h + (h * 0.125)

        im = frame.crop(crop_x1, crop_x2, crop_y1, crop_y2)

        cropx = im.shape[1]
        cropy = im.shape[0]

        if im.shape[0] < 64 or im.shape[1] < 64:
            return (None, [], duration_pp )

        start_pp = time.perf_counter()
        crop = self.preprocess(im)
        duration_pp += 1000 * (time.perf_counter() - start_pp)

        scale_x = cropx / 224.
        scale_y = cropy / 224.
        crop_info = (crop_x1, crop_y1, scale_x, scale_y)
        return (crop, crop_info, duration_pp )

    def early_exit(self, reason, start):
        print(reason)
        self.face = None
        duration = (time.perf_counter() - start) * 1000
        print(f"Took {duration:.2f}ms")
        return None, None

    def predict(self, frame):
        start = time.perf_counter()
        duration_fd = 0.0
        duration_pp = 0.0
        duration_model = 0.0
        duration_pnp = 0.0

        if self.face is None:
            start_fd = time.perf_counter()
            self.face = facedetection.detect_faces(frame.image, self.model, self.detection_threshold)
            duration_fd = 1000 * (time.perf_counter() - start_fd)
            if self.face is None:
                return self.early_exit("No faces found", start)

        crop, crop_info, duration_pp = self.cropFace(frame)

        #Early exit if crop fails, If the crop fails there's nothing to track
        if  crop is None:
            return self.early_exit("No valid crops", start)
        start_model = time.perf_counter()

        confidence, lms = landmarks.run(self.model, crop, crop_info)
        #Early exit if below confidence threshold
        if confidence < self.threshold:
            return self.early_exit("Confidence below threshold", start)

        eye_state = self.EyeTracker.get_eye_state(self.model, frame, lms)

        self.face_info.update((confidence, (lms, eye_state)), np.array(lms)[:, 0:2].mean(0))

        duration_model = 1000 * (time.perf_counter() - start_model)
        start_pnp = time.perf_counter()
        face_info = self.face_info

        if face_info.alive:
            face_info.success, face_info.quaternion, face_info.euler, face_info.pnp_error, face_info.pts_3d, face_info.lms = landmarks.estimate_depth(face_info, self.width, self.height)

            if face_info.success:
                face_info.adjust_3d()
                lms = face_info.lms[:, 0:2]
                y1, x1 = lms[0:66].min(0)
                y2, x2 = lms[0:66].max(0)
                self.face = [x1, y1, x2 - x1, y2 - y1]
                duration_pnp += 1000 * (time.perf_counter() - start_pnp)
                duration = (time.perf_counter() - start) * 1000
                print(f"Took {duration:.2f}ms (detect: {duration_fd:.2f}ms, crop: {duration_pp:.2f}ms, track: {duration_model:.2f}ms, 3D points: {duration_pnp:.2f}ms)")
                return face_info, self.face

        #Combined multiple failures into one catch all exit
        return self.early_exit("Face info not valid", start)
