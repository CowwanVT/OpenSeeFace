import numpy as np
import math
import featureExtractor
import emilianaFeatureExtractor

class FaceInfo():
    face_3d = np.array([
1
    def __init__(self, featureType):
        self.featureType = featureType



        self.reset()
        self.alive = False
        self.coord = None
        self.base_scale_v = self.face_3d[27:30, 1] - self.face_3d[28:31, 1]
        self.base_scale_h = np.abs(self.face_3d[[0, 36, 42], 0] - self.face_3d[[16, 39, 45], 0])
        self.fail_count = 0

    def reset(self):
        self.alive = False
        self.conf = None
        self.lms = None
        self.eye_state = None
        self.rotation = np.array([0.0, 0.0, 0.0], np.float32)
        self.translation = np.array([0.0, 0.0, 0.0], np.float32)
        self.success = None
        self.quaternion = None
        self.euler = None
        self.pts_3d = None
        self.eye_blink = None
        self.pnp_error = 0
        if self.featureType == 0:
            self.features = featureExtractor.FeatureExtractor()
        else:
            self.features = emilianaFeatureExtractor.FeatureExtractor()

        self.current_features = {}
        self.contour = np.zeros((21,3))
        self.update_contour()

    def update(self, result, coord):
        if result is None:
            self.reset()
        else:
            self.conf, (self.lms, self.eye_state) = result
            self.coord = coord
            self.alive = True

    def update_contour(self):
        self.contour = self.face_3d[[0,1,8,15,16,27,28,29,30,31,32,33,34,35]]

    def normalize_pts3d(self, pts_3d):
        # Calculate angle using nose
        pts_3d[:, 0:2] -= pts_3d[30, 0:2]
        alpha = (math.atan2(*((pts_3d[27, 0:2]) - pts_3d[30, 0:2])[::-1]) % (2*math.pi))
        cosalpha = math.cos(alpha-(math.pi/2))
        sinalpha = math.sin(alpha-(math.pi/2))

        R = np.matrix([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
        pts_3d[:, 0:2] = (pts_3d - pts_3d[30])[:, 0:2].dot(R) + pts_3d[30, 0:2]

        # Vertical scale
        pts_3d[:, 1] /= np.mean((pts_3d[27:30, 1] - pts_3d[28:31, 1]) / self.base_scale_v)

        # Horizontal scale
        pts_3d[:, 0] /= np.mean(np.abs(pts_3d[[0, 36, 42], 0] - pts_3d[[16, 39, 45], 0]) / self.base_scale_h)

        return pts_3d

    def adjust_3d(self):
        if self.conf < 0.4 or self.pnp_error > 300:
            return

        self.pts_3d = self.normalize_pts3d(self.pts_3d)
        self.current_features = self.features.update(self.pts_3d[:, 0:2])
        self.eye_blink = []
        self.eye_blink.append(1 - min(max(0, -self.current_features["eye_r"]), 1))
        self.eye_blink.append(1 - min(max(0, -self.current_features["eye_l"]), 1))
