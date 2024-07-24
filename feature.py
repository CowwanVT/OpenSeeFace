import maffs

class Feature():
    def __init__(self,decay=0.00001, curve=1, scaleType = 1,  spring = 1, friction = 0.5, mass = 2, standardDeviations = 3):
        self.minimum = None
        self.maximum = None
        self.span = None
        self.calibrate = None
        self.calibrated = False
        self.decay = decay
        self.last = 0.0
        self.curve = curve
        self.scaleType = scaleType
        self.stats = maffs.Stats()
        self.standardDeviations = standardDeviations
        self.mass = mass
        self.speed = 0
        self.old = 0
        self.spring = spring
        self.friction = friction
        self.calibrated = False

    def update(self, value, calibrate):
        if value is None:
            return 0
        self.calibrate = calibrate

        if self.scaleType == 1:
            normalizedValue = self.normalizeDoubleSided(value)
        else:
            normalizedValue = self.normalizeSingleSided(value)
            self.decaySpan()

        if self.calibrated == False:
            if self.calibrate == True:
                self.calibrated = True
            else:
                return 0
        new = normalizedValue

        momentum = self.speed * self.mass
        delta =  new - self.old
        force = delta * self.spring
        momentum = momentum + force
        self.speed = momentum/self.mass
        self.speed = self.speed * (1- self.friction)

        new = self.old + self.speed
        self.old = new

        return new

    def decaySpan(self):
        #The Min and Max decay slightly toward each other every frame allowing the range to change dynamically
        self.minimum = (self.minimum * (1 - self.decay)) + (self.maximum * self.decay)
        self.maximum = (self.maximum * (1 - self.decay)) + (self.minimum * self.decay)

    def normalizeSingleSided(self, value):
        if self.calibrate:

            if self.minimum is None or self.maximum is None:
                self.minimum = value - 0.00001
                self.maximum = value + 0.00001
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)
            self.span = self.maximum - self.minimum
            self.calibrated = True

        if self.calibrated:
            value = value - self.minimum
            normalizedValue = maffs.clamp(value/ self.span, 0, 1)
            return pow(normalizedValue, self.curve)
        else:
            return 0

    def normalizeDoubleSided(self, value):
        if self.calibrate:
            self.stats.update(value)
            self.calibrated = True
        if self.calibrated:

            stdDev = self.stats.getVariance()
            if stdDev == 0:
                return 0
            if value < self.stats.mean:
                return -pow(maffs.clamp(( self.stats.mean - value) / (stdDev * self.standardDeviations), 0, 1), self.curve)
            else:
                return pow(maffs.clamp((value - self.stats.mean) / (stdDev * self.standardDeviations), 0, 1), self.curve)
