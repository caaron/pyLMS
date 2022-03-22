import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import wave
from scipy.io.wavfile import read
import sys
#sys.path.append("../pyDSP")
from play import play

# these two function supplement your online measurement
def measure_x():
    # it produces input vector of size 3
    x = np.random.random(3)
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
farend_file = read('farend_short.wav')
d_file = read('mic_short.wav')
farend_data = np.array(farend_file[1]/32768.0,dtype=float)

#plt.figure()
d_data = np.array(d_file[1]/32768.0,dtype=float)
d_data = np.append(np.zeros(20),farend_data)
#plt.subplot(211)
#plt.plot(d_data)
d_data = d_data + np.random.random(len(d_data))/1000
#plt.subplot(212)
#plt.plot(d_data)
#plt.show()

if len(d_file[1]) > len(farend_file[1]):
    farend_data = np.append(farend_data,np.zeros(len(d_data)-len(farend_data)))
else:
    d_data = np.append(d_data,np.zeros(len(farend_data)-len(d_data)))


N = len(farend_data)
tailLength = 50  #int(.100 * 44100)
xDL = np.zeros(tailLength)
log_d = np.zeros(N)
log_y = np.zeros(N)
err = np.zeros(N)
filt = pa.filters.FilterLMS(tailLength, mu=.1, )
simul = False
for k in range(N):
    # measure input
    if simul:
        x = measure_x()
    else:
        xDL = delay(xDL)
        xDL[0] = farend_data[k]
    # predict new value
    y = filt.predict(xDL)
    if np.isnan(y):
        print("Output is NaN, probably use a smaller mu")
    # do the important stuff with prediction output
    pass
    # generate desired output from input
    if simul:
        d = generate_d(x)
    else:
        d = d_data[k]
    # update filter
    filt.adapt(d, xDL)
    # log values
    log_d[k] = d
    log_y[k] = y
    err[k] = y - d

### show results
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
plt.legend();
plt.subplot(223);
plt.title("Filter error")
plt.xlabel("samples - k")
plt.plot(log_d - log_y, "b", label="error")
plt.legend()
plt.subplot(224)
plt.title("Filter")
plt.xlabel("x")
plt.plot(filt.w, "b", label="error")
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
