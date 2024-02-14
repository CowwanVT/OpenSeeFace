import math

def clamp (value, minimum, maxium):
    return max(min(value,maxium),minimum)


def rotate(origin, point, a):
    x, y = point - origin
    a = -a

    qx = math.cos(a) * x - math.sin(a) * y

    qy = math.sin(a) * x + math.cos(a) * y

    qx+= origin[0]
    qy+= origin[1]

    return qx, qy

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
    return (int(x), int(y))

class Stats():
    def __init__(self):
        self.count = 0.
        self.mean = 0.
        self.M2 = 0.
        self.maximum = 0.
        self.minimum = 0.
        self.total = 0.

    def update(self,  new_value):
        self.total+= new_value
        self.maximum = max(new_value, self.maximum)
        self.minimum = min(new_value, self.minimum)
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += abs(delta * delta2)
        return

    def getMean(self):
        return self.mean

    def getVariance(self):
        if self.count > 2:
            return math.sqrt(self.M2/self.count)
        else:
            return 0.

    def getSampleVariance(self):
        if self.count > 2:
            return math.sqrt(self.M2/(self.count-1))
        else:
            return 0.
    def clamp(self, value):
        if self.count < 30:
            self.update(value)
            return value
        else:
            self.update(value)
            value = min( value, self.mean + (2* abs(self.getSampleVariance())))
            value = max( value, self.mean - (2* abs(self.getSampleVariance())))
            return value





