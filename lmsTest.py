# LMS filter test bench
# C. Aaron Hall
# Copyright 2022 by Garmin Ltd. or its subsidiaries.


import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import wave
from scipy.io.wavfile import read
import sys
import scipy.signal as sig
#sys.path.append("../pyDSP")
from play import play
from myLMS import myFilterLMS
from dtd import doubleTalkDetection

# these two function supplement your online measurement
def getInput():
    # it produces input vector of size 3
    x = np.random.random(1)
    return x


def generate_d(x):
    # generate system output
    d = 2 * x[0] + 1 * x[1] - 1.5 * x[2]
    return d

def delay(dl):
    for n in range(len(dl)-1,0,-1):
        dl[n] = dl[n-1]
    return dl


canc_file = wave.open('canc_offline.wav', 'wb')
Fs,farend_data = read('Hill.wav')
farend_data = farend_data[:Fs*3]

#dFs,d_file = read('mic_short.wav')
dFs = Fs
farend_data = np.array(farend_data/32768.0,dtype=float)

# use white noise to get maximum results
#farend_data = np.random.random(100000)
#if (dFs != Fs):
#    d_file = sig.resample(d_file,int(len(farend_data)/Fs))

#dFs,d_file = read('mic_short.wav')
h = np.array([0,0.5,0,0,0.3,0,0,0,0,0.1,0,0,0])
dFs = Fs

d_data = sig.lfilter(h,1,farend_data)

#plt.figure()
#d_data = np.array(d_file/32768.0,dtype=float)
# delay the farend data by 20 samples
#d_data = np.append(np.zeros(20),farend_data)
#plt.subplot(111)
#plt.stem(h,linefmt="C0:",label="h")
#markerline, stemlines, baseline = plt.stem(h*2,linefmt="C3--",markerfmt="D",label="estimate")
#markerline.set_markerfacecolor('none')
#plt.legend()
#plt.plot(d_data)
#plt.pause(.1)
#d_data = d_data + np.random.random(len(d_data))/10000
#plt.subplot(212)
#plt.plot(d_data)
#plt.show()

if len(d_data) > len(farend_data):
    farend_data = np.append(farend_data,np.zeros(len(d_data)-len(farend_data)))
else:
    d_data = np.append(d_data,np.zeros(len(farend_data)-len(d_data)))

N = len(farend_data)
tailLength = 50  #int(.100 * 44100)
xDL = np.zeros(tailLength)
log_d = np.zeros(N)
log_y = np.zeros(N)
err = np.zeros(N)
filt2 = pa.filters.FilterNLMS(tailLength, mu=1)
hh = np.append(h,np.zeros(tailLength-len(h)))
hh = hh + np.random.random(len(hh))/100
filt = myFilterLMS(tailLength, mu=.1, w="zeros")
d_data = np.append(d_data,np.zeros(tailLength))
dtdHistory = []

simul = False
for k in range(N):
    # measure input
    xDL = delay(xDL)
    xDL[0] = farend_data[k]
    # get desired output from input
    d = d_data[k]
    dDL = d_data[k:k+tailLength]
    # double talk detection
    dtdState = doubleTalkDetection(xDL,dDL)
    dtdHistory.append(dtdState)

    # predict new value
    y = filt.predict(xDL)
    #y = filt2.predict(xDL)
    if np.isnan(y):
        print("Output is NaN, probably use a smaller mu")
    # do the important stuff with prediction output
    pass
    if dtdState:
        pass
    else:
        # update filter
        filt.adapt(d, xDL)
        #filt2.adapt(d, xDL)
    # log values
    log_d[k] = d
    log_y[k] = y
    err[k] = y - d

### show results
xr = np.arange(len(log_d))/Fs
plt.figure()
plt.plot(d_data,label="SpkrOut")
plt.plot(farend_data,label="MicIn")
plt.plot(np.array(dtdHistory)*max(max(farend_data),max(d_data)),label="dtd")
plt.title("DTD")
plt.pause(.1)
plt.figure(figsize=(15, 9))
plt.subplot(221)
plt.title("Adaptation")
plt.xlabel("samples - k")
plt.plot(log_d, "b", label="d - target")
plt.plot(log_y, "g", label="y - output")
plt.legend()
plt.subplot(222)
plt.title("Filter error")
plt.xlabel("samples - k")
plt.plot(10 * np.log10((log_d - log_y) ** 2), "r", label="e - error [dB]")
plt.legend()
plt.grid()
plt.subplot(223)
plt.title("Filter error")
plt.xlabel("samples - k")
plt.plot(log_d - log_y, "b", label="error")
plt.legend()
plt.subplot(224)
plt.title("Filter")
plt.xlabel("taps")
#plt.stem(h, "b", label="System")
#plt.stem(filt2.w, "r", label="System Estimate")
plt.stem(h,linefmt="C0:",label="h")
markerline, stemlines, baseline = plt.stem(filt.w,linefmt="C3--",markerfmt="D",label="estimate")
markerline.set_markerfacecolor('none')
plt.grid()
plt.legend()

plt.tight_layout()
#plt.subplot(223);
#plt.draw()
plt.pause(.1)
#plt.pause(.1)
#plt.show()

### play results
#play(log_d,Fs=44100, nCh=1)
#play(farend_data,Fs=44100, nCh=1)
play(d_data,Fs=44100, nCh=1)
play(err,Fs=44100, nCh=1)


plt.pause(1)
