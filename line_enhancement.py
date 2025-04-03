import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import scipy.signal as sig
import scipy.fft as fft

FILTORDER = 100
M = 2

N = 10000
Fs = 48000
Fs2 = int(Fs/2)
t = np.arange(0,N)/Fs
v = np.random.normal(0, .01, N)
s = np.sin(2*np.pi*1000*t)
input = s + v
log_d = np.zeros(N)
log_e = np.zeros(N)
dlyline = np.zeros(FILTORDER)
inputdlyline = np.zeros(M)

filts = [
    (pa.filters.FilterLMS(FILTORDER, mu=.001,w="zeros"),"LMS"),
    #(pa.filters.FilterNLMS(FILTORDER, mu=.3),"NLMS"),
    #(pa.filters.FilterLMF(FILTORDER, mu=.1),"LMF"),
    #(pa.filters.FilterNLMF(FILTORDER, mu=.1),"NLMF"),
    #(pa.filters.FilterRLS(FILTORDER,.001,.0001),"RLS"),
    #(pa.filters.FilterAP(FILTORDER, mu=1.),"AP")
]
log_y = np.zeros((len(filts),N))

y = np.zeros(len(filts))

def delaysample(x,dl):
    tmp = dl[0:-1]        # throw out last sample
    result = np.insert(tmp, 0, x)           # insert new sample at the beginning
    return result

for n in range(N):
    inputdlyline = delaysample(input[n],inputdlyline)
    d = inputdlyline[-1]
    dlyline = delaysample(input[n],dlyline)

    for i in np.arange(len(filts)):
        y[i] = filts[i][0].predict(dlyline)
    #d = input[n]
    # update filter
    for i in np.arange(len(filts)):
        filts[i][0].adapt(d,dlyline)
    # log values
    log_d[n] = d
    log_e[n] = d - y[0]
    for i in np.arange(len(filts)):
        log_y[i][n] = y[i]

    #log_y[k] = y
    #log_y2[k] = y2


### show results
plt.figure(figsize=(15, 9))
plt.subplot(211);
plt.title("Adaptation");
plt.xlabel("samples - k")
plt.plot(log_d, "b", label="d - target")
for i in np.arange(len(filts)):
    plt.plot(log_y[i], label=f"y:{filts[i][1]}")
#plt.plot(log_y, "g", label="y - output");
plt.legend()
plt.grid()
plt.subplot(212);
plt.title("Filter error");
plt.xlabel("samples - k")
for i in np.arange(len(filts)):
    plt.plot(10 * np.log10((log_d - log_y[i]) ** 2), label=f"e:{filts[i][1]}")

#plt.plot(10 * np.log10((log_d - log_y) ** 2), label="e:LMS")
#plt.plot(10 * np.log10((log_d - log_y2) ** 2), label="e:AP")
plt.legend();
plt.tight_layout();
plt.grid()

plt.figure()
plt.subplot(211);
plt.plot(input, label="input")
plt.plot(s, label="noise free input")
plt.plot(log_y[0], label="output")
plt.legend()
plt.grid()
plt.subplot(212);
plt.plot(input, label="input")
plt.plot(s, label="noise free input")
plt.plot(log_e, label="error")
plt.legend()
plt.grid()

plt.figure()
F = fft.rfft(filts[0][0].w,1024)
w = np.arange(0, Fs2, Fs2 / len(F))
plt.plot(10 * np.log10(abs(F)), label="taps fft")

plt.pause(.1)
plt.show()