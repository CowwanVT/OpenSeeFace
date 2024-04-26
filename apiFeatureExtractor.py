import feature
import maffs

class APIfeatureExtractor():
    def __init__(self):
        self.jawOpen = feature.Feature(scaleType = 2, curve = 2)
        self.mouthX = feature.Feature(curve = 1, alpha=0.75)
        self.mouthFunnel = feature.Feature(scaleType = 2, curve = 2)
        self.mouthPucker = feature.Feature(curve = 2, alpha=0.5)
        self.mouthPressLipOpen = feature.Feature(curve = 2)
        self.eyeSquintR = feature.Feature(scaleType = 2, curve = 1)
        self.eyeSquintL = feature.Feature(scaleType = 2, curve = 1)
        self.mouthOpen = feature.Feature(scaleType = 2, curve = 0.5, alpha=0.01)
        self.eyeOpenLeft = feature.Feature(scaleType = 2, curve = 1, decay=0.01)
        self.eyeOpenRight = feature.Feature(scaleType = 2, curve = 1, decay=0.01)
        self.browLeftY = feature.Feature( curve = 2)
        self.browRightY = feature.Feature( curve = 2)
        self.mouthSmile= feature.Feature(curve = 2 )
        self.brows = feature.Feature( curve = 2)
        self.faceAngleX = feature.Feature( curve = 1.5, alpha=0.3, scaleType = 3)
        self.faceAngleY = feature.Feature( curve = 1.5, alpha=0.6, scaleType = 3)
        self.faceAngleZ = feature.Feature( curve = 1.5, alpha=0.3, scaleType = 3)
        self.eyeX = feature.Feature(  scaleType = 3)
        self.eyeY = feature.Feature(scaleType = 3)



    def update(self, pts, rotation):
        features = []

        f = maffs.euclideanDistance(maffs.average3d(pts[[41,42]]), maffs.average3d(pts[[36,39]]))
        features.append(["EyeSquintR", self.eyeSquintR.update(f)])

        f = -maffs.euclideanDistance(maffs.average3d(pts[[47,46]]), maffs.average3d(pts[[42,45]]))
        features.append(["EyeSquintL", self.eyeSquintL.update(f)])

        f =  maffs.euclideanDistance(maffs.average3d(pts[[58,62]]), pts[0]) - maffs.euclideanDistance(maffs.average3d(pts[[58,62]]), pts[16])
        features.append(["MouthX", self.mouthX.update(-f)])

        f = maffs.euclideanDistance(pts[33], pts[8])
        features.append(["JawOpen", self.jawOpen.update(f) ])

        f = maffs.euclideanDistance(pts[62], pts[58])
        features.append(["MouthPucker",  self.mouthPucker.update(f) ])

        f = maffs.euclideanDistance(maffs.average3d(pts[[59, 60, 61]]), maffs.average3d(pts[[65, 64, 63]]))
        mouth = self.mouthOpen.update(f)

        if mouth < 0.1:
            mouth = -0.1
        features.append(["MouthOpen", mouth])

        f = maffs.euclideanDistance(maffs.average3d(pts[[42,45]]), maffs.average3d(pts[[43,44]]))
        eyeLeft = self.eyeOpenLeft.update(f)


        f = maffs.euclideanDistance(maffs.average3d(pts[[36,39]]), maffs.average3d(pts[[37,38]]))
        eyeRight = self.eyeOpenLeft.update(f)
        eye = (eyeRight + eyeLeft)/2

        if eye < 0.1:
            eye = -0.1


        features.append(["EyeOpenRight", eye])
        features.append(["EyeOpenLeft", eye])

        f = (((pts[58][1] + pts[62][1])/2) - (pts[60][1])+( pts[60][1] - pts[64][1])*0.325-0.5)#/ norm_distance_y
        features.append(["MouthSmile", 0.5+ (self.mouthSmile.update(f)/2)])

        f = maffs.euclideanDistance(maffs.average3d(pts[[22,26]]), maffs.average3d(pts[[42,45]]))
        features.append(["BrowLeftY",0.5+self.browLeftY.update(f)/2])
        f = maffs.euclideanDistance(maffs.average3d(pts[[17,21]]), maffs.average3d(pts[[36,39]]))
        features.append(["BrowRightY",0.5+ self.browRightY.update(f)/2])


        f = maffs.euclideanDistance(maffs.average3d(pts[[17,21,22,26]]), maffs.average3d(pts[[36,39,42,45]]))
        features.append(["Brows", 0.5+ self.brows.update(f)/2])

        f = -((pts[66][0] + pts[67][0])/2)
        bothEyesX = self.eyeX.update(f)
        features.append(["EyeRightX", bothEyesX])
        features.append(["EyeLeftX", bothEyesX])

        f = -((pts[66][1] + pts[67][1])/2)
        bothEyesY = self.eyeY.update(f)
        features.append(["EyeRightY", bothEyesY])
        features.append(["EyeLeftY", bothEyesY])

        f = pts[0][2] - pts[16][2]
        features.append(["FaceAngleX", 30 * self.faceAngleX.update(f)])
        f = pts[30][1] - pts[33][1]
        features.append(["FaceAngleY", 10 * self.faceAngleY.update(f)])
        f = rotation[0]
        features.append(["FaceAngleZ", 30 * self.faceAngleZ.update(f)])
        #features.append(["FaceAngleZ", 0])

        return(features)
