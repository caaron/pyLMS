
import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sig
#from EC import doPlots
from scipy.io.wavfile import read,write

mMicisJustDelay = True
#canc_file = wave.open('canc_offline.wav', 'wb')
Fs,farend_alldata = read('farend.wav')
floatFEdata = np.array(farend_alldata/32768.0,dtype=float)
if mMicisJustDelay:
    nDly = 10
    z = np.zeros(nDly)
    d = np.array(floatFEdata[0:-nDly],copy=True)
    floatMICdata = np.concatenate((z, d))
else:
    mic_alldata = read('mic.wav')
    floatMICdata = np.array(mic_alldata[1] / 32768.0, dtype=float)
    #floatMICdata = np.random.random(len(floatFEdata))

CHUNK = int(.02*Fs)  # number of data points to read at a time
FILTERLENGTH = int(.016 * Fs)

fig, axes = plt.subplots(2,1)
t=np.arange(CHUNK)
farend_frame = floatFEdata[:CHUNK]
mic_frame = floatMICdata[:CHUNK]
axes[0].plot(t,farend_frame,t,mic_frame)
axes[1].plot(floatFEdata)
plt.pause(.001)
mic_saved = np.zeros(CHUNK)
maxcArr = [np.empty(0), np.empty(0)]
maxc = 0
m = 100
dlyEstimate = np.arange(nDly-5,nDly+5)
erle = np.zeros((len(dlyEstimate),len(floatFEdata)))
idx = 0
axes[0].clear()
axes[1].clear()
frameByFrame = False
if frameByFrame:
    for d in dlyEstimate:

        if frameByFrame:
            start = 0
            farend_frame = floatFEdata[:CHUNK]
            mic_frame = floatMICdata[:CHUNK]
            frames = 0
            while len(farend_frame) > 0 and len(mic_frame) > 0:
                # LMS
                filtOutput = np.zeros(CHUNK)
                error = np.zeros(CHUNK)

                axes[0].clear()
                t = np.arange(start,start+CHUNK)
                axes[0].plot(t, farend_frame, t, mic_saved)
                axes[0].plot(t, mic_frame - mic_saved)
                axes[0].legend(["farend", "mic", "diff"])
                axes[1].clear()
                axes[1].plot(maxcArr[0])
                axes[1].plot(maxcArr[1])
                axes[1].plot(maxc * np.ones(CHUNK))

                #plt.show()
                plt.pause(.001)
                plt.pause(.1)

                frames += 1
                start = frames * CHUNK
                farend_frame = floatFEdata[start:start + CHUNK]
                mic_frame = floatMICdata[start:start + CHUNK]
        else:
            # do whole file processing
            x1 = floatFEdata[0:-d]
            x = np.concatenate((np.zeros(d), x1))
            err = (x - floatMICdata) + 1e-10
            erle[idx] = 10*np.log10(abs(err))
            axes[0].plot(erle[idx])
            c = sig.correlate(x,floatMICdata)
            axes[1].plot(c)
            plt.pause(.001)
            plt.pause(1)
            idx += 1

erle2 = np.zeros((len(dlyEstimate),len(floatFEdata)))
dlyEstimate = np.arange(nDly-9,nDly-3)
idx = 0
axes[0].clear()
axes[1].clear()
floatMICdata = sig.lfilter([0,0,0,0,0,0,.6,0,0,0,-.5,0,0,.4],1,floatFEdata)
#floatMICdata = sig.lfilter([0,0,0,0,0,0,.6],1,floatFEdata)
for d in dlyEstimate:
    m = np.max(np.abs(floatFEdata))
    x1 = floatFEdata[0:-d]
    x = np.concatenate((np.zeros(d), x1)) * m
    err = (x - floatMICdata) + 1e-10
    erle2[idx] = 10*np.log10(abs(err**2))
    axes[0].plot(erle2[idx])
    c = sig.correlate(x, floatMICdata)
    axes[1].plot(c)
    plt.pause(.001)
    plt.pause(1)
    idx += 1

plt.pause(.001)
plt.pause(.1)
rng = np.random.default_rng()
x = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
sig_noise = x + rng.standard_normal(len(x))
corr = sig.correlate(sig_noise, np.ones(128), mode='same') / 128
clock = np.arange(64, len(x), 128)
fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
#ax_orig.plot(x)
ax_orig.plot(floatMICdata)

ax_orig.plot(clock, x[clock], 'ro')
ax_orig.set_title('Original signal')
ax_noise.plot(sig_noise)
ax_noise.set_title('Signal with noise')
ax_corr.plot(corr)
ax_corr.plot(clock, corr[clock], 'ro')
ax_corr.axhline(0.5, ls=':')
ax_corr.set_title('Cross-correlated with rectangular pulse')
ax_orig.margins(0, 0.1)
fig.tight_layout()
plt.pause(.1)
plt.show()