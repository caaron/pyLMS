import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
#import collections
from ringbuff import RingBuffer
import scipy.signal as sig
from ringbuff import historyBuffer

CHUNK = 1024
frames = 0
#dlyLine = collections.deque(maxlen= 3 * CHUNK)
#dlyLine2 = collections.deque(maxlen= 3 * CHUNK)
dlyLine = historyBuffer(2 * CHUNK)
#while not dlyLine.is_full:
#    dlyLine.append(0)

fig, axes = plt.subplots(2,1)

Fs,farend_alldata = read('farend.wav')
floatFEdata = np.array(farend_alldata/32768.0,dtype=float)
Fs = 16000
t = np.arange(3*Fs)/Fs
floatFEdata = (32767 * np.sin(2*np.pi*100*t)).astype(np.int32)
nDly = 50

z = np.zeros(nDly)
d = 1 * np.array(floatFEdata[0:-nDly],copy=True)
floatMICdata = np.concatenate((z, d))
h = np.array([0,0,0,0,0,0,0,0,0,.7,0,0,0,.3,0,0,0,.1])
#h = np.array([0,0,0,0,0,0,0,0,0,.7])
floatMICdata = sig.lfilter(h,1,floatFEdata)
mic_frame = floatMICdata[:CHUNK]
farend_frame = floatFEdata[:CHUNK]

from rms import rms

nTrials = 10
nAvg = 10
dlyAvg = np.zeros(nAvg)
#ampAvg = np.zeros((nTrials,nAvg))
ampAvg = np.zeros(nTrials)
ErrDlyLine = [[] for i in range(nTrials)]
dlyStart = 8

yHistory = np.zeros(len(farend_alldata))

while len(mic_frame) == CHUNK:
    dlyLine.push(farend_frame)

    micRMS = rms(mic_frame)
    spkrRMS = rms(farend_frame)
    rmsRatio = micRMS / spkrRMS
    ampAvg = np.insert(ampAvg[:-1], 0, rmsRatio)
    idxDly = 0
    for dly in range(dlyStart,dlyStart+nAvg):
        y = (ampAvg.mean() * dlyLine[dly:dly + CHUNK])
        err = y - mic_frame[::-1]
        errRMS = 20*np.log10(rms(err))
        ErrDlyLine[idxDly].append(errRMS)
        idxDly += 1

        if False:
            offset = 4
            axes.clear()
            #axes[0].plot(fer)
            axes.plot(mic_frame[::-1])
        #    axes[0].plot(dlyLine[CHUNK:(CHUNK+CHUNK)])
            axes.plot(y)
            axes.legend(['mic_frame',f'dlyline {dly}'])
            plt.grid()
            #plt.figtext(.5,0, f'c:{mc} at {mci}')
            if False:
                axes[1][0].clear()
                #axes[1].plot(farend_frame)
                #axes[1].plot(dlyLine[(offset):(offset+CHUNK)])
                #axes[1].plot(dlyLine[:CHUNK])
                axes[1][0].plot(c)
                #axes[1].plot(dlyLine[(3*offset):(3*offset+CHUNK)])
                #axes[1].legend(['farend','mic[offset]','mic'])
                dplot = np.hstack([np.zeros(intDly-1), np.array(mc), np.zeros(len(c)-intDly+1)])
                axes[1][0].plot(dplot)

                axes[0][1].clear()
                axes[0][1].plot(mic_frame[::-1])
                axes[0][1].plot(y)
                axes[0][1].plot(err)
                axes[0][1].set_ylim(-1,1)
                axes[0][1].legend(['mic_frame', 'y', 'error'])

                axes[1][1].clear()
                eplot = ErrDlyLine[:]
                axes[1][1].plot(eplot)
                plt.xlabel(f"intDly:{intDly + 1} so delay {CHUNK - intDly} samples")

            plt.pause(.1)
            plt.pause(.1)

    yHistory[start:start + CHUNK] = y
    frames += 1
    start = frames*CHUNK
    mic_frame = floatMICdata[start:start + CHUNK]
    farend_frame = floatFEdata[start:start + CHUNK]

idxDly = 0
for i in range(dlyStart,dlyStart+nAvg):
    axes[0].plot(ErrDlyLine[idxDly],label=f"delay{i}")
    idxDly += 1
axes[0].legend()
axes[1].plot(yHistory, label=f"delay{i}")
plt.pause(.1)
plt.pause(.1)
plt.show()