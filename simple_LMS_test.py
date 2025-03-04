import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import scipy.signal as sig

FILTORDER = 7
if False:
    # creation of x and d
    N = 1700
    x = np.random.random((N, FILTORDER))
    v = np.random.normal(0, 1, N) * 0.001
    h = [2,.1,-4,.5,-.2,-.4,.3]
    d = np.dot(x,h) + v
    #d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v

    # identification
    f1 = pa.filters.FilterLMS(mu=0.4, n=FILTORDER)
    f2 = pa.filters.FilterNLMS(mu=0.4, n=FILTORDER)
    y, e, w = f1.run(d, x)
    y2, e2, w2 = f2.run(d, x)

    # show results
    plt.figure(figsize=(12.5,9))
    plt.subplot(211);plt.title("Adaptation");plt.xlabel("Number of iteration [-]")
    plt.plot(d,"b", label="d:target")
    plt.plot(y,"g", label="y:output")
    plt.plot(v,"r", label="noise")
    plt.xlim(0, N)
    plt.legend()
    plt.grid()

    plt.subplot(212); plt.title("Filter error"); plt.xlabel("Number of iteration [-]")
    plt.plot(pa.misc.logSE(e), label="LMS ERLE");plt.legend()
    plt.plot(pa.misc.logSE(e2), label="NLMS ERLE");plt.legend()
    plt.xlim(0, N)
    plt.tight_layout()
    plt.grid()



    plt.pause(.1)
    plt.pause(.1)
    #print("And the resulting coefficients are: {}".format(w[-1]))

# these two function supplement your online measurment
def measure_x(idx):
    # it produces input vector of size 3
    #x = np.random.random(3)
    #x = np.random.random(FILTORDER)
    x = mic[idx:idx+FILTORDER]
    return x

h = np.array([2,1,-1.5])
#h = np.array([0,0,0,0,.8,0,.4,0,0,-.2])
h = np.random.random(FILTORDER)
def measure_d(x):
    # meausure system output
    #d = 2 * x[0] + 1 * x[1] - 1.5 * x[2]
    if len(x) != len(h):
        print("error!!")
    acc = np.dot(x,h)
    v = np.random.normal(0, 1) * .001
    return acc + v

N = 1000
spkr = np.random.random(N+FILTORDER)
v = np.random.normal(0, 1, N+FILTORDER) * .1
mic = spkr + v
u = pa.input_from_history(mic,FILTORDER)[:-1]
#x1 = np.empty(0)
#for n in range(N):
#    x1 = np.append(x1,np.random.random(1))
#d2 = sig.lfilter(h,1,x1)
log_d = np.zeros(N)
log_y = np.zeros(N)
filt = pa.filters.FilterLMS(FILTORDER, mu=.4)
#filt = pa.filters.FilterAP(3, mu=1.)
for k in range(N):
    # measure input
    x = measure_x(k)
    if not np.array_equal(x,u[k]):
        print("input incorrect")
    #x = x1[k]
    # predict new value
    y = filt.predict(x)
    # do the important stuff with prediction output
    pass
    # measure output
    d = measure_d(x)
    # update filter
    filt.adapt(d, x)
    # log values
    log_d[k] = d
    log_y[k] = y


### show results
plt.figure(figsize=(15, 9))
plt.subplot(211);
plt.title("Adaptation");
plt.xlabel("samples - k")
plt.plot(log_d, "b", label="d - target")
plt.plot(log_y, "g", label="y - output");
plt.legend()
plt.subplot(212);
plt.title("Filter error");
plt.xlabel("samples - k")
plt.plot(10 * np.log10((log_d - log_y) ** 2), "r", label="e - error [dB]")
plt.legend();
plt.tight_layout();
plt.figure()
plt.stem(h, label="h");
plt.stem(filt.w, label="w");

#plt.figure()
#plt.plot(spkr, label="spkr")
#plt.plot(mic, label="mic")
#plt.legend()
#plt.grid()

plt.pause(.1)
plt.show()