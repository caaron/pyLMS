
import numpy as np
import matplotlib.pylab as plt

def rms(x):
    return np.sqrt(np.dot(x,x))

def dB(x):
    return 20*np.log10(x)

def doubleTalkDetection(ref,mic,param=6):
    refPower = dB(rms(ref))
    micpwr = dB(rms(mic))

    # param is a generic
    # interpreted here as if mic power > (ref power - param dB), return true
    # so, mic power must be param dB less than the ref power to adapt
    if micpwr > (refPower - param):
        return True
    else:
        return False
