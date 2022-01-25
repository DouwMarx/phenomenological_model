import numpy as np
from scipy.signal import hilbert


class Inner(object):
    def __init__(self):
        pass

    def somefunc(self):
        return np.sin(np.linspace(0, 1, 100))

class Outer(object):
    def __init__(self):
        pass

    def env(self):
        sig = Inner().somefunc()
        return hilbert(sig)

a = Outer()
print(a.env())

b = Outer()
print(b.env())
