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
ErrDlyLine = historyBuffer(2 * CHUNK,initialValue=-100)
#while not dlyLine.is_full:
#    dlyLine.append(0)

fig, axes = plt.subplots(2,2)

Fs,farend_alldata = read('farend.wav')
floatFEdata = np.array(farend_alldata/32768.0,dtype=float)
Fs = 16000
t = np.arange(3*Fs)/Fs
#floatFEdata = (32767 * np.sin(2*np.pi*10*t)).astype(np.int32)
nDly = 10
z = np.zeros(nDly)
d = 1 * np.array(floatFEdata[0:-nDly],copy=True)
floatMICdata = np.concatenate((z, d))
mic_frame = floatMICdata[:CHUNK]
farend_frame = floatFEdata[:CHUNK]

class circBuff:
    def __init__(self,size=10):
        self.size = size
        self.q = collections.deque(maxlen= size)

    def push_frame(self,data):
        for x in data:
            self.q.append(x)
    def pop_frame(self,length):
        result = self.q.pop()
        for i in np.arange(length-1):
            result = np.append(result,self.q.pop())
        return result

    def pop_frame_4_plotting(self,length=0):
        if length<=0:
            L = len(self.q)
        else:
            L = length
        x = self.pop_frame(L)
        return x[::-1]

from rms import rms

nAvg = 10
dlyAvg = np.zeros(nAvg)
ampAvg = np.zeros(nAvg)

while len(mic_frame) > 0:
#    for x in farend_frame:
        #dlyLine.appendleft(x)
#        dlyLine.append(x)
        #dlyLine2.append(x)
    dlyLine.push(farend_frame)

    frm = dlyLine[:CHUNK]
    fer = farend_frame[::-1]
    if not np.array_equal(fer,frm):
        print("WTF?!?")
    c = sig.correlate(mic_frame[::-1], frm)
    mci = np.argmax(c)
    mc = max(c)
    dlyAvg = np.insert(dlyAvg[:-1], 0, mci)
    # ampAvg = np.concatenate(mc,ampAvg[:-1])       # update delayline of indexes
    micRMS = rms(mic_frame)
    spkrRMS = rms(farend_frame)
    rmsRatio = micRMS / spkrRMS
    ampAvg = np.insert(ampAvg[:-1], 0, rmsRatio)
    intDly = int(np.ceil(np.mean(dlyAvg)))
    y = (ampAvg.mean() * dlyLine[CHUNK - intDly:CHUNK - intDly + CHUNK])
    err = y - mic_frame[::-1]
    errRMS = 20*np.log10(rms(err))
    ErrDlyLine.push(errRMS)

    offset = 4
    axes[0][0].clear()
    #axes[0].plot(fer)
    axes[0][0].plot(mic_frame[::-1])
#    axes[0].plot(dlyLine[CHUNK:(CHUNK+CHUNK)])
    axes[0][0].plot(frm)
    axes[0][0].legend(['mic_frame','dlyline'])
    #plt.figtext(.5,0, f'c:{mc} at {mci}')
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

    frames += 1
    start = frames*CHUNK
    mic_frame = floatMICdata[start:start + CHUNK]
    farend_frame = floatFEdata[start:start + CHUNK]
