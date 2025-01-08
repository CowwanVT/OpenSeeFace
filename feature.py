import maffs

class Feature():
    def __init__(self,decay=0.00001, curve=1, scaleType = 1,  spring = 1, friction = 0.5, mass = 2, standardDeviations = 3, statisticalSmoothing = False, originSpring = 0.0):
        self.minimum = None
        self.maximum = None
        self.span = None
        self.calibrated = False
        self.decay = decay
        self.curve = curve
        self.scaleType = scaleType
        self.valueStats = maffs.Stats()
        self.speedStats = maffs.Stats()
        self.accelerationStats = maffs.Stats()
        self.standardDeviations = standardDeviations
        self.mass = mass
        self.speed = 0
        self.old = 0
        self.spring = spring
        self.friction = friction
        self.confidenceThrehold = 0.85
        self.raw = 0.0
        self.previousRaw = 0.0
        self.statisticalSmoothing = statisticalSmoothing
        self.originSpring = originSpring

    def update(self, value, confidence):
        if value is None:
            return 0

        if confidence <= self.confidenceThrehold and not self.calibrated:
            return 0

        if confidence > self.confidenceThrehold:
            self.calibrate(value)

        if self.scaleType == 1:
            normalizedValue = self.normalizeDoubleSided(value)
        else:
            normalizedValue = self.normalizeSingleSided(value)

        smoothedValue = self.smoothMotion(normalizedValue)
        return smoothedValue

    def smoothMotion(self, value):
        delta =  value - self.old
        acceleration = delta - self.speed
        #acceleration = self.accelerationStats.clamp(acceleration)
        delta = self.speed + acceleration
        if self.statisticalSmoothing:
            delta = self.speedStats.clamp(delta)

        force = delta * self.spring
        if self.originSpring != 0:
            force = force - (value * self.originSpring)
        momentum = self.speed * self.mass

        momentum = momentum + force
        self.speed = momentum/self.mass
        self.speed = self.speed * (1- self.friction)
        value = self.old + self.speed
        self.old = value
        return value


    def calibrate(self, value):

        value = self.valueStats.clamp(value)


        if self.scaleType == 2:
            if self.minimum is None or self.maximum is None:
                self.minimum = value - 0.00001
                self.maximum = value + 0.00001
            else:
                self.decaySpan
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)
            self.span = self.maximum - self.minimum

        self.calibrated = True


    def decaySpan(self):
        #The Min and Max decay slightly toward each other every frame allowing the range to change dynamically
        self.minimum = (self.minimum * (1 - self.decay)) + (self.maximum * self.decay)
        self.maximum = (self.maximum * (1 - self.decay)) + (self.minimum * self.decay)

    def normalizeSingleSided(self, value):

        value = value - self.minimum
        normalizedValue = maffs.clamp(value/ self.span, 0, 1)
        return pow(normalizedValue, self.curve)


    def normalizeDoubleSided(self, value):
        stdDev = self.valueStats.getVariance()
        if stdDev == 0:
            return 0
        if value < self.valueStats.mean:
            return -pow(maffs.clamp(( self.valueStats.mean - value) / (stdDev * self.standardDeviations), 0, 1), self.curve)
        else:
            return pow(maffs.clamp((value - self.valueStats.mean) / (stdDev * self.standardDeviations), 0, 1), self.curve)
