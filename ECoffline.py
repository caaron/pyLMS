import struct

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave

import scipy

import recorder
import padasip as pa
from scipy.io.wavfile import read

#fig, (ax1, ax2) = plt.subplots(2,1)

mMicisJustDelay = True
fig, ax1 = plt.subplots()
#fe_file = wave.open('farend.wav', 'rb')
#mic_file = wave.open('mic.wav','rb')
canc_file = wave.open('canc_offline.wav', 'wb')
farend_alldata = read('farend.wav')
floatFEdata = np.array(farend_alldata[1]/32768.0,dtype=float)
if mMicisJustDelay:
    nDly = 10
    floatMICdata = np.array(np.zeros(nDly), floatFEdata[0:-nDly])
else:
    mic_alldata = read('mic.wav')
    floatMICdata = np.array(mic_alldata[1] / 32768.0, dtype=float)



RATE = 48000 # time resolution of the recording device (Hz)
CHUNK = int(.02*RATE)  # number of data points to read at a time
FILTERLENGTH = int(.016 * RATE)

canc_file.setsampwidth(2)
canc_file.setnchannels(1)
canc_file.setframerate(farend_alldata[0])

lms = pa.filters.FilterLMS(n=FILTERLENGTH, mu=0.001, w="zeros")

primerFrames = 0
mPlotdata = True

dlyline = np.zeros(FILTERLENGTH)



def delaysample(x,dl):
    tmp = dl[0:-1]        # throw out last sample
    result = np.insert(tmp, 0, x)           # insert new sample at the beginning
    return result


def delayfilter(x):
    y = scipy.signal.lfilter([0,0,0,1],1,x)
    return y


def fixed2float(x):
    y = np.array(x,dtype=float)
    return y
    ll = int(len(x)/2)
    y = np.zeros(ll)
    for i in range(0,ll):
        idx = int(i*2)
        y[i] = float((x[idx+1] * 32768) + x[idx])/32768
    return y
    idx = 0
    for smp in x:
        y[idx] = smp/32768.0
        idx += 1
    return y

def float2fixed(x):
    ll = int(len(x)*2)
    y = np.zeros(ll)
    y = np.zeros(len(x))
    idx = 0
    for smp in x:
        y[idx] = int(smp * 32768.0)
        idx += 1
    return y

maxerr = 0
frames = 0
start = frames*CHUNK
farend_frame = floatFEdata[start:start+CHUNK]
#mic_frame = floatMICdata[start:start+CHUNK]
mic_frame = delayfilter(farend_frame)

try:
    while len(farend_frame) > 0 and len(mic_frame) > 0:
        # LMS
        filtOutput = np.zeros(CHUNK)
        error = np.zeros(CHUNK)

        for idx in range(0, CHUNK):
            x = farend_frame[idx]
            dlyline = delaysample(x, dlyline)
            y = lms.predict(dlyline)
            d = mic_frame[idx]
            e = y - d
            if abs(e) > maxerr:
                maxerr = abs(e)
#            if abs(e) > .2:
            lms.adapt(e, dlyline)
            filtOutput[idx] = y
            error[idx] = e

        if mPlotdata:
            ax1.clear()
            t = np.arange(frames*CHUNK,(frames+1)*CHUNK)
            d1 = ax1.plot(t,farend_frame,t,mic_frame,t,filtOutput,t,error)
            #ax1.set_ylim(-32768, 32767)
            ax1.legend(["farend","mic","estimate","error"])
            plt.show()
            plt.pause(.001)

        yfixed = float2fixed(error)
        canc_file.writeframes(yfixed)

        frames += 1
        start = frames * CHUNK
        farend_frame = floatFEdata[start:start + CHUNK]
        #mic_frame = floatMICdata[start:start + CHUNK]
        mic_frame = delayfilter(farend_frame)

except EOFError:
    print('Hello user it is EOF exception, please enter something and run me again')
except KeyboardInterrupt:
    print('quitting...')

farend_alldata.close()
canc_file.close()
mic_alldata.close()
