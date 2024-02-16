import feature
import maffs

class APIfeatureExtractor():
    def __init__(self):
        self.jawOpen = feature.Feature(scaleType = 2, curve = 2)
        self.mouthX = feature.Feature(curve = 2)
        self.mouthFunnel = feature.Feature(scaleType = 2, curve = 2)
        self.mouthPucker = feature.Feature(curve = 2)
        self.mouthPressLipOpen = feature.Feature(curve = 2)
        self.eyeSquintR = feature.Feature(scaleType = 2, curve = 2)
        self.eyeSquintL = feature.Feature(scaleType = 2, curve = 2)

    def update(self, pts):
        features = []
        norm_distance_y = pts[27, 1]-pts[8, 1]

        f = (pts[41][1] + pts[42][1])/2 - (pts[36][1] + pts[39][1])/2
        features.append(["EyeSquintR", self.eyeSquintR.update(f)])

        f = (pts[47][1] + pts[46][1])/2 - (pts[42][1] + pts[45][1])/2
        features.append(["EyeSquintL", self.eyeSquintL.update(f)])

        f_pts = maffs.align_points(pts[27], pts[33], pts[[33, 50, 55]])
        f = (pts[50][0] + pts[55][0] )/2 - pts[33][0]
        features.append(["MouthX", self.mouthX.update(f)])

        f = ( pts[33][1] - pts[8][1]) / norm_distance_y
        features.append(["JawOpen", self.jawOpen.update(f) ])

        f = pts[62][0] - pts[58][0]
        features.append(["MouthPucker",  self.mouthPucker.update(f) ])



        return(features)
