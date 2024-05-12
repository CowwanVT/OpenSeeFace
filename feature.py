import maffs


#I redid a lot of this so it worked well for me
#idk if it'll work well for other people
class Feature():
    def __init__(self, alpha=0.1, decay=0.00001, curve=1, scaleType = 1):
        self.min = None
        self.max = None
        self.alpha = alpha
        self.decay = decay
        self.last = 0.0
        self.curve = curve
        self.scaleType = scaleType
        self.stats = maffs.Stats()

    def update(self, x):
        new = self.update_state(x)
        self.last = self.last * self.alpha + new * (1 - self.alpha)
        return self.last

    def update_state(self, x):
        if self.min is None or self.max is None:
            self.min = x - 0.00001
            self.max = x + 0.00001
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        center = (self.min+self.max)/2

        #The Min and Max decay slightly toward each other every frame allowing the range to change dynamically
        self.min = (self.min * (1 - self.decay)) + (self.max * self.decay)
        self.max = (self.max * (1 - self.decay)) + (self.min * self.decay)

        if self.scaleType == 1:
            #Returns a value between -1 and 1 in relation to the maximum range
            if x < center:
                return -pow(maffs.clamp((x - center) / (self.min - center), 0, 1), self.curve)
            elif x > center:
                return pow(maffs.clamp((x - center) / (self.max - center), 0, 1), self.curve)
            return 0
        if self.scaleType == 2:
            #Returns a value between 0 and 1 in relation to the maximum range
                return pow(maffs.clamp((x - self.min) / (self.max - self.min), 0, 1), self.curve)
        if self.scaleType ==3:
            self.stats.update(x)
            #Returns a value between -1 and 1 in relation to the maximum range, 0 is the mean
            if x < self.stats.mean:
                return -pow(maffs.clamp((x - self.stats.mean) / (self.min - self.stats.mean), 0, 1), self.curve)
            else:
                return pow(maffs.clamp((x - self.stats.mean) / (self.max - self.stats.mean), 0, 1), self.curve)
        return 0

