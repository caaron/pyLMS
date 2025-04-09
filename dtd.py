
import numpy as np
import matplotlib.pylab as plt

def rms(x):
    return np.sqrt(np.dot(x,x))

def dB(x):
    return 20*np.log10(x)

MinRefPower = -6

def doubleTalkDetection(ref,mic,param=6):
    refPower = dB(rms(ref))
    micpwr = dB(rms(mic))

    # param is a generic
    # interpreted here as if mic power > (ref power - param dB), return true
    # so, mic power must be param dB less than the ref power to adapt

    if refPower > MinRefPower:       ## this needs to be an adaptive threshold.
        if micpwr > (refPower - param):
            return True
        else:
            return False
    else:
        return False


def corrDTD(ref, mic, param=6):
    refPower = dB(rms(ref))
    micpwr = dB(rms(mic))

    corr1 = sig.correlate(ref,mic)

    # param is a generic
    # interpreted here as if mic power > (ref power - param dB), return true
    # so, mic power must be param dB less than the ref power to adapt
    if micpwr > (refPower - param):
        return True
    else:
        return False
