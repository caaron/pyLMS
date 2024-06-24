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
#floatFEdata = (32767 * np.sin(2*np.pi*100*t)).astype(np.int32)
nDly = 50

z = np.zeros(nDly)
d = 1 * np.array(floatFEdata[0:-nDly],copy=True)
floatMICdata = np.concatenate((z, d)) + (np.random.random(np.size(floatFEdata))/1e6)
h = np.array([0,0,0,0,0,0,0,0,0,.7,0,0,0,.3,0,0,0,.1])
#h = np.array([0,0,0,0,0,0,0,0,0,.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-.5])
#h = np.array([0,0,0,0,0,0,0,0,0,.7])
#h = np.array([0,0,0,0,0,0,0,0,0,1.0])
#floatMICdata = sig.lfilter(h,1,floatFEdata)
#floatMICdata = sig.lfilter(h,1,floatFEdata) + (np.random.random(np.size(floatFEdata))/1e6)
mic_frame = floatMICdata[:CHUNK]
farend_frame = floatFEdata[:CHUNK]

from rms import rms

nTrials = 10
nAvg = 10
dlyAvg = np.zeros(nAvg)
#ampAvg = np.zeros((nTrials,nAvg))
ampAvg = np.zeros(nTrials)
#ErrDlyLine = [[] for i in range(nTrials)]
dlyStart = 8

yHistory = np.zeros(len(floatMICdata))
errHistory = np.zeros(len(floatMICdata))
errLogHistory = np.empty(0)
peakvsRMS = np.empty(0)
RMSidxs = np.empty(0)
start = 0

while len(mic_frame) == CHUNK:
    dlyLine.push(farend_frame)

    micRMS = rms(mic_frame)
    spkrRMS = rms(farend_frame)
    rmsRatio = micRMS / spkrRMS
    peakRatio = max(abs(mic_frame)) / max(abs(farend_frame))
    ampAvg = np.insert(ampAvg[:-1], 0, rmsRatio)
    peak_Errs = np.zeros(dlyLine.size - CHUNK) # MSE for every delay possibility using peak amplitude matching
    RMS_Errs = np.zeros(dlyLine.size - CHUNK) # MSE for every delay possibility using peak amplitude matching
    best_peak = 100
    best_peak_idx = -1
    best_RMS = 100
    best_RMS_idx = -1
    idxDly = 0

    for dly in range(0,dlyLine.size - CHUNK):
        yPeak = (peakRatio * dlyLine[dly:dly + CHUNK])
        if frames < nAvg:
            yRMS = (rmsRatio * dlyLine[dly:dly + CHUNK])
        else:
            yRMS = (ampAvg.mean() * dlyLine[dly:dly + CHUNK])
        errPeak = yPeak - mic_frame[::-1]
        errRMS = yRMS - mic_frame[::-1]
        errPeaklog = 20*np.log10(rms(errPeak))
        errRMSlog = 20*np.log10(rms(errRMS))
 #       ErrDlyLine[idxDly].append(errRMS)
        # now fill in peak_Errs with each delay's MSE, then pick the best as the actual y
        peak_Errs[dly] = errPeaklog
        RMS_Errs[dly] = errRMSlog
        if errPeaklog < best_peak:
            best_peak_idx = dly
            best_rms_y = yRMS
            best_log = errPeaklog
        if errRMSlog < best_RMS:
            best_RMS = errRMSlog
            best_RMS_idx = dly
            best_peak_y = yPeak
            best_log = errRMSlog
        idxDly += 1

    if False:
        offset = 4
        axes[0].clear()
        #axes[0].plot(fer)
        axes[0].plot(mic_frame)
    #    axes[0].plot(dlyLine[CHUNK:(CHUNK+CHUNK)])
        axes[0].plot(best_peak_y[::-1])
        axes[0].plot(farend_frame)
        axes[0].legend(['mic_frame',f'best_peak_y','farend_frame'])
        plt.grid()
        axes[1].clear()
        axes[1].plot(best_peak_y[::-1] - mic_frame)
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

    RMSidxs = np.append(RMSidxs,best_peak_idx)

    if RMS_Errs.min() < peak_Errs.min():
        peakvsRMS = np.append(peakvsRMS,1)      # 1 means RMS was lower MSE than peak
        y = best_rms_y[::-1]
    else:
        peakvsRMS = np.append(peakvsRMS, 0)     # 0 means peak was better
        y = best_peak_y[::-1]

    y = best_peak_y[::-1]
    err = y - mic_frame
    errHistory[start:start + CHUNK] = err
    erle = 20*np.log10(rms(err)/rms(mic_frame))
    errLogHistory = np.append(errLogHistory, erle)
    #errHistory[start:start + CHUNK] = errRMS
    yHistory[start:start + CHUNK] = y

    frames += 1
    start = frames*CHUNK
    mic_frame = floatMICdata[start:start + CHUNK]
    farend_frame = floatFEdata[start:start + CHUNK]

avgERLE = np.mean(errLogHistory)
axes[0].plot(errLogHistory,label=f"errLogHistory")
axes[0].plot(np.ones_like(errLogHistory) * avgERLE,label=f"avgERLE")
#axes[0].plot(RMSidxs,label=f"bestIDxs")
axes[0].legend()
axes[1].plot(yHistory, label=f"y")
axes[1].plot(floatMICdata, label=f"mic")
axes[1].plot(errHistory, label=f"err")
axes[1].legend()
plt.pause(.1)
plt.pause(.1)
plt.show()