import os
os.environ["OMP_NUM_THREADS"] = str(1)
import numpy as np
import onnxruntime
import cv2
cv2.setNumThreads(6)
import time
import eyes
import landmarks
import face

def prepareImageForModel(frame):
    mean = np.float32(np.array([-2.1179, -2.0357, -1.8044]))
    std = np.float32(np.array([0.0171, 0.0175, 0.0174]))
    targetimageSize = [224, 224]

    image = cv2.resize(frame, targetimageSize, interpolation=cv2.INTER_CUBIC)
    image = image * std + mean

    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0,3,1,2))
    return image

class Models():
    models = [
        "lm_model0_opt.onnx",
        "lm_model1_opt.onnx",
        "lm_model2_opt.onnx",
        "lm_model3_opt.onnx",
        "lm_model4_opt.onnx"]

    optimizationLevel = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    def __init__(self, args):
        self.detectionThreshold = args.detection_threshold

        modelBasePath = os.path.join(os.path.dirname(__file__), os.path.join("models"))
        providersList = onnxruntime.capi._pybind_state.get_available_providers()
        faceDetectModel = os.path.join(modelBasePath,"mnv3_detection_opt.onnx")
        gazeModel = os.path.join(modelBasePath, "mnv3_gaze32_split_opt.onnx")
        landmarksModel = os.path.join(modelBasePath, self.models[args.model])
        optimizationLevel = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1

        options.intra_op_num_threads =  args.threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = optimizationLevel
        options.log_severity_level = 3

        self.landmarks = onnxruntime.InferenceSession(landmarksModel, sess_options=options, providers=providersList)
        self.gazeTracker = onnxruntime.InferenceSession(gazeModel, sess_options=options, providers=providersList)

        options.intra_op_num_threads =  2
        self.faceDetection = onnxruntime.InferenceSession(faceDetectModel, sess_options=options, providers=providersList)

    def detectFaces(self, frame):

        image = prepareImageForModel(frame)

        outputs, maxpool = self.faceDetection.run([], {'input': image})
        outputs = outputs[0]
        maxpool = maxpool[0]
        mask = outputs[0] == maxpool[0]
        mask2 = outputs[1] > 0.15
        outputs[0] = outputs[0] * mask.astype(int) * mask2.astype(int)

        faceLocation = np.argmax(outputs[0].flatten())
        x = faceLocation % 56
        y = faceLocation // 56

        if outputs[0, y, x] < self.detectionThreshold:
            return None
        r = outputs[1, y, x] * 112.
        x = int((x * 4) - r) * (frame.shape[1] / 224.)
        y = int((y * 4) - r) * (frame.shape[0] / 224.)
        w = int(r * 2) * (frame.shape[1] / 224.)
        h = int(r * 2) * (frame.shape[0] / 224.)
        results = np.array([x,y,w,h], dtype = np.int32)
        return results

    def detectLandmarks(self, crop, crop_info):

        output = self.landmarks.run([], {"input": crop})[0][0]
        confidence, lms = landmarks.landmarks(output, crop_info)
        return (confidence, lms)

class Tracker():
    def __init__(self, args):

        self.EyeTracker = eyes.EyeTracker()
        self.model = Models(args)

        self.threshold = args.threshold
        self.face = None
        self.face_info = face.FaceInfo(args.feature_type)

    def cropFace(self, frame):
        duration_pp = 0.0
        x,y,w,h = self.face

        crop_x1 = int( x - (w * 0.1))
        crop_y1 = int( y - (h * 0.125))
        crop_x2 = int( x + w + (w * 0.1))
        crop_y2 = int( y + h + (h * 0.125))

        im = frame.crop(crop_x1, crop_x2, crop_y1, crop_y2)
        if im.shape[0] < 64 or im.shape[1] < 64:
            return (None, [], duration_pp )

        start_pp = time.perf_counter()
        crop = prepareImageForModel(im)
        duration_pp += 1000 * (time.perf_counter() - start_pp)

        scale_x = im.shape[1] / 224.
        scale_y = im.shape[0] / 224.
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
            self.face =  self.model.detectFaces(frame.image)

            duration_fd = 1000 * (time.perf_counter() - start_fd)

            if self.face is None:
                return self.early_exit("No faces found", start)

        crop, crop_info, duration_pp = self.cropFace(frame)

        if  crop is None:
            return self.early_exit("No valid crops", start)

        start_model = time.perf_counter()

        confidence, lms = self.model.detectLandmarks(crop, crop_info)
        if confidence < self.threshold:
            return self.early_exit("Confidence below threshold", start)

        eye_state = self.EyeTracker.get_eye_state(self.model, frame, lms)
        self.face_info.update((confidence, (lms, eye_state)), np.array(lms)[:, 0:2].mean(0))

        duration_model = 1000 * (time.perf_counter() - start_model)
        start_pnp = time.perf_counter()

        face_info = self.face_info

        if not face_info.alive:
            return self.early_exit("Face info not valid", start)

        face_info = landmarks.estimate_depth(face_info, frame.width, frame.height)

        if not face_info.success:
            return self.early_exit("Face info not valid", start)

        lms = face_info.lms[:, 0:2]
        y1, x1 = lms[0:66].min(0)
        y2, x2 = lms[0:66].max(0)
        self.face = np.array([x1, y1, x2 - x1, y2 - y1], dtype = np.int32)
        duration_pnp = 1000 * (time.perf_counter() - start_pnp)
        duration = (time.perf_counter() - start) * 1000
        print(f"Took {duration:.2f}ms, detect: {duration_fd:.2f}ms,", end = " ")
        print(f"crop: {duration_pp:.2f}ms, track: {duration_model:.2f}ms", end = " ")
        print(f"3D points: {duration_pnp:.2f}ms")
        return face_info, self.face
