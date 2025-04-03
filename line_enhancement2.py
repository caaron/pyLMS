import matplotlib.pylab as plt
import padasip as pa
import numpy as np
from scipy.io import wavfile
#from scipy.fftpack import fft
import scipy.fft as fft
import scipy.signal as sig


def zs(a):
    """ 1d data z-score """
    a -= a.mean()
    return a / a.std()


# constants
FILENAME = "farend_short.wav"
useFileIO = True

SAMPLERATE = 48000
n = 500 # filter size
D = 2 # signal delay
N = 10000
Fs = 48000
Fs2 = int(Fs/2)

# open and process source data
if useFileIO:
    Fs, data = wavfile.read(FILENAME)
    s = data.copy()
    s = s.astype("float32")
    s = s/32768.0
    #s = zs(s) / 10
    N = len(s)
# contaminated with noise
    v = np.sin(2*np.pi*1000/99*np.arange(N) + 10.1 * np.sin(2*np.pi/110*np.arange(N)))
    #v = np.random.normal(0, .1, N)
    d = s + v
else:
    t = np.arange(0,N)/Fs
    v = np.random.normal(0, .5, N)
    v = .3 * np.sin(2*np.pi*6027*t)
    v = sig.chirp(t,10,N/Fs,22000) * .2
    s = np.sin(2*np.pi*1000*t)
    d = s + v


plt.figure(figsize=(12.5,6))

plt.subplot(211)
plt.plot(v)
plt.title("Noise")

plt.subplot(212)
plt.plot(d)
plt.title("Contaminated data (our observation)")

plt.tight_layout()
#plt.show()


plt.pause(.1)
plt.pause(.1)
#plt.show()

# prepare data for simulation
x = pa.input_from_history(d, n)
x = x[:-D]
d = d[n+D-1:]
#y = y[n+D-1:]
#q = q[n+D-1:]

# create filter and filter
f = pa.filters.FilterNLMS(n=n, mu=0.01, w="zeros")
yp, e, w = f.run(d, x)
wavfile.write("distorted.wav", Fs, d)
wavfile.write("enhanced.wav", Fs, e)

plt.figure(figsize=(12.5,6))

plt.subplot(211)
plt.plot(d,label='d')
plt.plot(s,label='s')
#plt.plot(yp,label='y')
plt.plot(e,label='e')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(10 * np.log10(e ** 2), label=f"error**2")
plt.legend()
plt.grid()

plt.tight_layout()

plt.figure(figsize=(12.5,6))
plt.subplot(211)
plt.stem(f.w,label='w')
plt.legend()

plt.subplot(212)
F = fft.rfft(f.w,1024)
w = np.arange(0, Fs2, Fs2 / len(F))
plt.plot(w,10 * np.log10(abs(F)), label="taps fft")
plt.legend()
plt.grid()

plt.pause(.1)
plt.pause(.1)
