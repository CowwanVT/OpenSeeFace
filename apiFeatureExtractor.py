import feature
import maffs

class APIfeatureExtractor():
    def __init__(self):
        self.jawOpen = feature.Feature(scaleType = 2, curve = 2)
        self.mouthX = feature.Feature(curve = 2, alpha=0.5)
        self.mouthFunnel = feature.Feature(scaleType = 2, curve = 2)
        self.mouthPucker = feature.Feature(curve = 2, alpha=0.5)
        self.mouthPressLipOpen = feature.Feature(curve = 2)
        self.eyeSquintR = feature.Feature(scaleType = 2, curve = 1)
        self.eyeSquintL = feature.Feature(scaleType = 2, curve = 1)

    def update(self, pts):
        features = []

        f = maffs.euclideanDistance(maffs.average3d(pts[[41,42]]), maffs.average3d(pts[[36,39]]))
        features.append(["EyeSquintR", self.eyeSquintR.update(f)])

        f = -maffs.euclideanDistance(maffs.average3d(pts[[47,46]]), maffs.average3d(pts[[42,45]]))
        features.append(["EyeSquintL", self.eyeSquintL.update(f)])

        f = maffs.euclideanDistance(maffs.average3d(pts[[58,62]]), pts[0]) - maffs.euclideanDistance(maffs.average3d(pts[[58,62]]), pts[16])
        features.append(["MouthX", self.mouthX.update(f)])

        f = maffs.euclideanDistance(pts[33], pts[8])
        features.append(["JawOpen", self.jawOpen.update(f) ])

        f = maffs.euclideanDistance(pts[62], pts[58])
        features.append(["MouthPucker",  self.mouthPucker.update(f) ])

        return(features)
