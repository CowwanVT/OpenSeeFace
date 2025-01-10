import feature
import maffs

class APIfeatureExtractor():
    def __init__(self):
        self.jawOpen = feature.Feature(scaleType = 2, curve = 0.5,  spring = 1, friction = 0.5, mass = 2)
        self.mouthX = feature.Feature(curve = 1, scaleType = 1,  spring = 0.5, friction = 0.33, mass = 1, originSpring = 0.05)
        self.mouthOpen = feature.Feature(scaleType = 2, curve = 1, decay=0.0001,  spring = 1, friction = 0.6, mass = 0.6, originSpring = 0.1)
        self.mouthSmile= feature.Feature(curve = 2,  spring = 0.75, friction = 0.5, mass = 1, originSpring = 0.1)
        self.mouthPucker = feature.Feature(curve = 1.5,  spring = 0.75, friction = 0.5, mass = 1)

        self.eyeSquintR = feature.Feature(scaleType = 2, curve = 1,  spring = 1, friction = 0.5, mass = 2)
        self.eyeSquintL = feature.Feature(scaleType = 2, curve = 1,  spring = 1, friction = 0.5, mass = 2)
        self.eyeOpenLeft = feature.Feature(scaleType = 2, curve = 1, decay=0.0001,  spring = 1, friction = 0.5, mass = 2)
        self.eyeOpenRight = feature.Feature(scaleType = 2, curve = 1, decay=0.0001,  spring = 1, friction = 0.5, mass = 2)

        self.eyeX = feature.Feature(curve = 1, scaleType = 1,  spring = 0.8, friction = 0.5, mass = 1, originSpring = 0.1)
        self.eyeY = feature.Feature(curve = 1, scaleType = 1,  spring = 0.8, friction = 0.5, mass = 1.5, originSpring = 0.1)

        self.browLeftY = feature.Feature( scaleType = 2, curve = 1,  spring = 1, friction = 0.5, mass = 2)
        self.browRightY = feature.Feature( scaleType = 2, curve = 1,  spring = 1, friction = 0.5, mass = 2)
        self.brows = feature.Feature( scaleType = 2, curve = 1,  spring = 1, friction = 0.5, mass = 2)

        self.faceAngleX = feature.Feature( curve = 1.1, scaleType = 1, spring = 0.33, friction = 0.33, mass = 1, originSpring = 0.1)
        self.faceAngleY = feature.Feature( curve = 1.1, scaleType = 1, spring = 0.33, friction = 0.33, mass = 2, originSpring = 0.1)
        self.faceAngleZ = feature.Feature( curve = 1.1, scaleType = 1, spring = 0.33, friction = 0.33, mass = 2, originSpring = 0.3)

        self.facePositionX = feature.Feature( curve = 1, scaleType = 1, spring = 0.33, friction = 0.33, mass = 3, originSpring = 0.1)
        self.facePositionY = feature.Feature( curve = 1, scaleType = 1, spring = 0.33, friction = 0.33, mass = 3, originSpring = 0.2)
        self.facePositionZ = feature.Feature( curve = 1, scaleType = 1, spring = 0.33, friction = 0.33, mass = 3, originSpring = 0.2)

        self.previousEyeX = 0
        self.previousEyeY = 0

        self.calibrated = False

    def update(self, pts, rotation, facePosition, confidence):

        if confidence > 0.85:
            calibrate = True
            self.calibrated = True
        else:
            calibrate = False
        features = []

        if self.calibrated:

            f = maffs.euclideanDistance(maffs.average3d(pts[[41,42]]), maffs.average3d(pts[[36,39]]))
            features.append(["EyeSquintR", self.eyeSquintR.update(f, confidence)])

            f = -maffs.euclideanDistance(maffs.average3d(pts[[47,46]]), maffs.average3d(pts[[42,45]]))
            features.append(["EyeSquintL", self.eyeSquintL.update(f, confidence)])

            f =  maffs.euclideanDistance(maffs.average3d(pts[[58,62]]), pts[0]) - maffs.euclideanDistance(maffs.average3d(pts[[58,62]]), pts[16])
            features.append(["MouthX", self.mouthX.update(-f, confidence)])

            f = maffs.euclideanDistance(pts[33], pts[8])
            features.append(["JawOpen", self.jawOpen.update(f, confidence) ])

            f = maffs.euclideanDistance(pts[62], pts[58])
            features.append(["MouthPucker",  self.mouthPucker.update(f, confidence) ])

            f = maffs.euclideanDistance(maffs.average3d(pts[[59, 60, 61]]), maffs.average3d(pts[[65, 64, 63]]))
            mouth = self.mouthOpen.update(f, confidence)

            if mouth < 0.15:
                mouth = -0.1
            features.append(["MouthOpen", mouth])

            f = maffs.euclideanDistance(maffs.average3d(pts[[42,45]]), maffs.average3d(pts[[43,44]]))
            eyeLeft = self.eyeOpenLeft.update(f, confidence)

            f = maffs.euclideanDistance(maffs.average3d(pts[[36,39]]), maffs.average3d(pts[[37,38]]))
            eyeRight = self.eyeOpenLeft.update(f, confidence)
            eye = (eyeRight + eyeLeft)/2

            if eye < 0.15:
                eye = -0.1

            features.append(["EyeOpenRight", eye])
            features.append(["EyeOpenLeft", eye])

            f = (((pts[58][1] + pts[62][1])/2) - (pts[60][1])+( pts[60][1] - pts[64][1])*0.325-0.5)#/ norm_distance_y
            features.append(["MouthSmile", 0.5+ (self.mouthSmile.update(f, confidence)/2)])

            f = maffs.euclideanDistance(maffs.average3d(pts[[22,26]]), maffs.average3d(pts[[42,45]]))
            features.append(["BrowLeftY",self.browLeftY.update(f, confidence)])

            f = maffs.euclideanDistance(maffs.average3d(pts[[17,21]]), maffs.average3d(pts[[36,39]]))
            features.append(["BrowRightY", self.browRightY.update(f, confidence)])


            f = maffs.euclideanDistance(maffs.average3d(pts[[17,21,22,26]]), maffs.average3d(pts[[36,39,42,45]]))
            features.append(["Brows", self.brows.update(f, confidence)])

            if eye > 0.3333:

                f = -((pts[66][0] + pts[67][0])/2)
                bothEyesX = self.eyeX.update(f, confidence)
                features.append(["EyeRightX", bothEyesX])
                features.append(["EyeLeftX", bothEyesX])

                f = -((pts[66][1] + pts[67][1])/2)
                bothEyesY = self.eyeY.update(f, confidence)
                features.append(["EyeRightY", bothEyesY])
                features.append(["EyeLeftY", bothEyesY])
                self.previousEyeX = bothEyesX
                self.previousEyeY = bothEyesY
            else:
                features.append(["EyeRightX", self.previousEyeX])
                features.append(["EyeLeftX", self.previousEyeX])
                features.append(["EyeRightY", self.previousEyeY])
                features.append(["EyeLeftY", self.previousEyeY])


            f = rotation[0]
            xAxis =  30 * self.faceAngleX.update(f, confidence)
            features.append(["FaceAngleX",xAxis])

            f = -rotation[1]
            features.append(["FaceAngleY", 20 * self.faceAngleY.update(f, confidence)])

            f = rotation[2]
            features.append(["FaceAngleZ", (30 * self.faceAngleZ.update(f, confidence)) ])

            f = facePosition[0]
            features.append(["FacePositionX", -self.facePositionX.update(f, confidence)*5])

            f = facePosition[1]
            features.append(["FacePositionY", -self.facePositionY.update(f, confidence)*5])

            f = facePosition[2]
            features.append(["FacePositionZ", -self.facePositionZ.update(f, confidence)*5])

        return(features)
