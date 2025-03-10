import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import scipy.signal as sig

FILTORDER = 25

# these two function supplement your online measurment
def measure_x(idx):
    # it produces input vector of size 3
    #x = np.random.random(3)
    #x = np.random.random(FILTORDER)
    x = mic[idx:idx+FILTORDER]
    return x

h = np.random.random(FILTORDER)
m = np.sum(np.abs(h))
if m > 1.0:
    h /= np.sum(np.abs(h))
    m = np.sum(np.abs(h))
    print(f"rescaled h, sum(abs(h))={m}")

def measure_d(x):
    # meausure system output
    #d = 2 * x[0] + 1 * x[1] - 1.5 * x[2]
    if len(x) != len(h):
        print("error!!")
    acc = np.dot(x,h)
    v = np.random.normal(0, 1) * .001
    return acc + v

N = 10000
spkr = np.random.random(N+FILTORDER)
v = np.random.normal(0, 1, N+FILTORDER) * .1
mic = spkr + v
u = pa.input_from_history(mic,FILTORDER)[:-1]
log_d = np.zeros(N)
filts = [
    (pa.filters.FilterLMS(FILTORDER, mu=.1),"LMS"),
    (pa.filters.FilterNLMS(FILTORDER, mu=.1),"NLMS"),
    (pa.filters.FilterAP(FILTORDER, mu=1.),"AP")
]
log_y = np.zeros((len(filts),N))

y = np.zeros(len(filts))

for k in range(N):
    # measure input
    x = measure_x(k)
    if not np.array_equal(x,u[k]):
        print("input incorrect")
    #x = x1[k]
    # predict new value
    for i in np.arange(len(filts)):
        y[i] = filts[i][0].predict(x)
    #y = filt.predict(x)
    #y2 = filt2.predict(x)
    # do the important stuff with prediction output
    pass
    # measure output
    d = measure_d(x)
    # update filter
    for i in np.arange(len(filts)):
        filts[i][0].adapt(d,x)
#    filt.adapt(d, x)
#    filt2.adapt(d, x)
    # log values
    log_d[k] = d
    for i in np.arange(len(filts)):
        log_y[i][k] = y[i]

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
plt.stem(h, label="h");
for i in np.arange(len(filts)):
    plt.stem(filts[i][0].w, label=f"w:{filts[i][1]}");
#plt.stem(filt.w, label="w");

#plt.figure()
#plt.plot(spkr, label="spkr")
#plt.plot(mic, label="mic")
#plt.legend()
#plt.grid()

plt.pause(.1)
plt.show()