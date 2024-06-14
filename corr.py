
import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sig
from scipy.io.wavfile import read,write

mMicisJustDelay = True
#canc_file = wave.open('canc_offline.wav', 'wb')
Fs,farend_alldata = read('farend.wav')
floatFEdata = np.array(farend_alldata/32768.0,dtype=float)
if mMicisJustDelay:
    nDly = 10
    z = np.mean(floatFEdata) * np.ones(nDly)
    d = np.array(floatFEdata[0:-nDly],copy=True)
    floatMICdata = np.concatenate((z, d))
else:
    mic_alldata = read('mic.wav')
    floatMICdata = np.array(mic_alldata[1] / 32768.0, dtype=float)

CHUNK = int(.02*Fs)  # number of data points to read at a time
FILTERLENGTH = int(.016 * Fs)
start = 0
farend_frame = floatFEdata[:CHUNK]
mic_frame = floatMICdata[:CHUNK]
frames = 0

fig, axes = plt.subplots(2,1)
t=np.arange(CHUNK)
axes[0].plot(t,farend_frame,t,mic_frame)
axes[1].plot(np.zeros(CHUNK))
plt.pause(.001)
mic_saved = np.zeros(CHUNK)
maxcArr = [np.empty(0), np.empty(0)]
maxc = 0


for dly in np.arange(25,500,25):
    z = np.mean(floatFEdata) * np.ones(dly)
    d = np.array(floatFEdata[0:-dly],copy=True)
    floatMICdata = np.concatenate((z, d))
    start = 10000
    farend_frame = floatFEdata[start:start + CHUNK]
    mic_frame = floatMICdata[start:start+CHUNK]
    c = sig.correlate(farend_frame, mic_frame)
    c1 = sig.correlate(farend_frame, mic_frame, mode='same')
    mc = np.argmax(c)
    mc1 = np.argmax(c1)
    axes[0].clear()
    t = np.arange(start, start + CHUNK)
    axes[0].plot(t, farend_frame, t, mic_frame)
    axes[0].legend(["farend", "mic"])
    axes[1].clear()
    axes[1].plot(c)
    axes[1].plot(c1)
    print(f"dly:{dly},c:{mc}, c1:{mc1}-->CHUNK{CHUNK}-mc{mc}={CHUNK-mc}:::CHUNK{CHUNK}-mc{mc1}={CHUNK-mc1}")
    plt.pause(.1)
    plt.pause(.1)

while len(farend_frame) > 0 and len(mic_frame) > 0:
    # LMS
    filtOutput = np.zeros(CHUNK)
    error = np.zeros(CHUNK)

#    for idx in range(0, CHUNK):
#        x = farend_frame[idx]
#        dlyline = delaysample(x, dlyline)
#        y = lms.predict(dlyline)
#        d = mic_frame[idx]


    if frames >= 1:
        zpMic = floatMICdata[start-CHUNK:start+CHUNK]
        maxcArr = [np.empty(0), np.empty(0)]
        maxc = 0
        maxcIdx = 0
        for dly in np.arange(3 * len(farend_frame)):
            dlydMic = zpMic[dly:dly+CHUNK]
            c = sig.correlate(farend_frame,dlydMic)
            cacc = sum(c)
            maxcArr[0] = np.append(maxcArr[0],cacc)
            maxcArr[1] = np.append(maxcArr[1],max(c))
            if cacc > maxc:
                maxc = cacc
                maxcIdx = dly
                mic_saved = dlydMic

            if dly >= 880 and True:
                axes[0].clear()
                t = np.arange(start, start + CHUNK)
                axes[0].plot(t, farend_frame, t, dlydMic)
                axes[0].legend(["farend", "mic"])
                axes[1].clear()
                axes[1].plot(c)
                plt.pause(.1)
                plt.pause(.1)

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