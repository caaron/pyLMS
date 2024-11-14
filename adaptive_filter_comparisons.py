import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import scipy.signal as sig
import scipy.fft as fft
#from corr import floatFEdata


# these two function supplement your online measurment
def measure_x():
    # it produces input vector of size 3
    x = np.random.random(3)
    return x

def measure_d(x):
    # meausure system output
    d = 2*x[0] + 1*x[1] - 1.5*x[2]
    return d

import wave
from scipy.io.wavfile import read
import math

mMicisJustDelay = True
farend_alldata = read('farend.wav')
floatFEdata = np.array(farend_alldata[1][:len(farend_alldata[1])>>5]/32768.0,dtype=float)
floatFEdata = np.random.random(32000)

if mMicisJustDelay:
    nDly = 5
    z = np.zeros(nDly)
    d = np.array(floatFEdata[0:-nDly],copy=True)
    floatMICdata = np.concatenate((z, d))
else:
    mic_alldata = read('mic.wav')
    floatMICdata = np.array(mic_alldata[1] / 32768.0, dtype=float)


def generateX(b,a,title,Fs=48000):
    x1 = np.random.random(Fs*2)
    x = sig.lfilter(b,a,x1)
    fig, axes = plt.subplots(2,1)
    w, h = sig.freqz(b,a,fs=Fs)
    axes[0].plot(w,10*np.log10(abs(h)))
    F = fft.rfft(x)
    w = np.arange(0,Fs2,Fs2/len(F))
    axes[0].set_title(title)
    axes[1].plot(w,10*np.log10(abs(F)))
    plt.pause(.1)
    plt.pause(.1)
    return x

def generateD(x,b,a,title):
    d = sig.lfilter(b,a,x)
    return d

def run_simulation(input,desired,filt,filtTitle):
    N = len(input)
    log_d = np.zeros(N)
    log_y = np.zeros(N)


    for i in range(N-FILTORDER):
        # measure input
        #x = measure_x()
        x = input[i:i+FILTORDER]
        # predict new value
        y = filt.predict(x)
        # do the important stuff with prediction output
        pass
        # get the desired input
        d = desired[i]
        # update filter
        filt.adapt(d, x)
        # log values
        log_d[i] = d
        if math.isnan(y):
            log_y[i] = 1e-10
        else:
            log_y[i] = y

    ### show results
    #plt.figure(figsize=(15,9))
    plt.figure()
    plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
    plt.plot(log_d,"b", label="d - target")
    plt.plot(log_y,"g", label="y - output");plt.legend()
    plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
    err = 10*np.log10(1e-20+(log_d-log_y)**2)
    plt.plot(err,"r", label="e - error [dB]")
    avgPerformance = np.ones(len(err))*np.mean(err[int(.9*len(err)):])
    plt.plot(avgPerformance,"b", label="avg error [dB]")
    plt.legend(); plt.tight_layout();
    plt.pause(.1)
    plt.show()
    plt.pause(.1)



Fs = 48000
Fs2 = Fs>>1
b1 = Fs2*.25
b2 = Fs2*.5
b,a = sig.butter(11,[b1/Fs, b2/Fs], 'bandpass')

FILTORDER = 10
filts = [
    [pa.filters.FilterLMS(FILTORDER, mu=.1), 'LMS'],
    [pa.filters.FilterAP(n=FILTORDER, order=5, mu=0.5, ifc=0.001, w="random"), 'Affine Projection'],
         ]
#v = np.random.normal(0, 1, N) * 1e-5
#d1 = 2*x + 0.1*x - 4*x + 0.5*x + v
h1 = [2, .1, -4, .5]
#sim1
x = generateX(b,a,f'butterworth-11 [{b1}, {b2}]')
d = generateD(x,h1,1,'d??')
for f,title in filts:
    run_simulation(x,d,f,title)

