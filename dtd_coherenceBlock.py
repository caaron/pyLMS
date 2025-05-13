import numpy as np
from scipy.io.wavfile import read
import scipy.signal as sig
from scipy.signal import coherence, windows
import matplotlib.pyplot as plt
import threading
import csv
from queue import Queue


class BlockCoherenceDTD:
    def __init__(self, fs=16000, block_size=1024, hop_size=512, threshold=0.6, num_threads=4, log_file="dtd_log.csv"):
        self.fs = fs
        self.block_size = block_size
        self.hop_size = hop_size
        self.threshold = threshold
        self.window = windows.hann(block_size)
        self.num_threads = num_threads
        self.log_file = log_file
        self.task_queue = Queue()
        self.log_state = []
        self.log_coh = []
        self.log_timestamp = []

    def process_block(self, block_near, block_far, start_time):
        # Compute coherence
        f, Cxy = coherence(block_near, block_far, fs=self.fs, nperseg=self.block_size, window=self.window)
        avg_coherence = np.mean(Cxy)

        # Detect double talk
        double_talk = avg_coherence > self.threshold
        result = (start_time, avg_coherence, double_talk)

        # Log result
        #with open(self.log_file, mode='a', newline='') as file:
        #    writer = csv.writer(file)
        #    writer.writerow(result)
        self.log_coh.append([avg_coherence]*len(block_near))
        self.log_state.append([1 if double_talk else 0]*len(block_near))
        self.log_timestamp.append([start_time]*len(block_near))

        # Print to console
        print(
            f"Time: {start_time:.3f} s | Coherence: {avg_coherence:.4f} | Double Talk: {'YES' if double_talk else 'NO'}")

    def worker(self):
        while not self.task_queue.empty():
            block_near, block_far, start_time = self.task_queue.get()
            try:
                self.process_block(block_near, block_far, start_time)
            finally:
                self.task_queue.task_done()

    def detect(self, far_end, near_end, doPlots=False):
        # Read wav files
#        far_end, fs1 = sf.read(far_end_file, dtype='float32')
#        near_end, fs2 = sf.read(near_end_file, dtype='float32')

#        if fs1 != fs2:
#            raise ValueError("Sample rates do not match.")

 #       self.fs = fs1  # Update fs in case it's different


        time_axis = []
    #    doPlots = False
        if doPlots:
            f = plt.figure(figsize=(15, 6))
            plt.subplot(211)
            plt.plot(far_end, label="farend")
            plt.plot(near_end, label="nearend")
            plt.legend()
            plt.pause(.1)
            plt.pause(.1)
            f,Pxx = sig.welch(far_end,fs=self.fs,nperseg=self.block_size,window=self.window)
            _, Pyy = sig.welch(near_end, fs=self.fs, nperseg=self.block_size, window=self.window)
            _, Pxy = sig.csd(far_end, near_end, fs=self.fs, nperseg=self.block_size, window=self.window)
            plt.subplot(212)
            plt.cla()
            plt.plot(f, Pxx, label='Pxx')
            plt.plot(f, Pyy, label='Pyy')
            plt.plot(f, Pxy, label='Pxy')
            plt.legend()
            plt.title(f'PSDs')
            plt.pause(.1)
            plt.pause(.1)

        frame = 0
        coherence_values = np.zeros(min_len)
        log_DT = np.zeros(min_len)

        # Process in blocks
        for start in range(0, min_len - self.block_size + 1, self.hop_size):
            block_far = far_end[start:start + self.block_size]
            block_near = near_end[start:start + self.block_size]

            # Compute coherence
            f, Cxy = coherence(block_near, block_far, fs=self.fs, nperseg=self.block_size, window=self.window)
            _, Pxx = sig.welch(block_far,fs=self.fs,nperseg=self.block_size,window=self.window)
            _, Pyy = sig.welch(block_near, fs=self.fs, nperseg=self.block_size, window=self.window)
            _, Pxy = sig.csd(block_far, block_near, fs=self.fs, nperseg=self.block_size, window=self.window)
            PxyDiff = Pxx - Pyy
            PxyDiffScore = np.sum(np.abs(PxyDiff))/block_far.size
            correlation = sig.correlate(block_near, block_far, mode="full")
            lags = sig.correlation_lags(block_near.size, block_far.size, mode="full")
            lag = lags[np.argmax(correlation)]
            norm_factor = np.sqrt(np.sum(block_near**2) * np.sum(block_far**2))
            ncc = correlation / norm_factor
            nccScore = np.max(ncc)

#            if np.argmax(correlation) != np.argmax(np.abs(correlation)):
#                print(f"correlation max different at frame {frame}")
            avg_coherence = np.mean(np.abs(Pxy))
            avg_coherence2 = np.mean(np.abs(Pxy[:int(8000*self.block_size/self.fs/2)]))
#            Cxy2 = np.abs(Pxy)**2 / Pxx / Pyy
#            avg_coherence2 = np.mean(Cxy[:int(8000*self.block_size/self.fs/2)])

            # Detect double talk
            double_talk = avg_coherence > self.threshold
            #for n in range(self.hop_size):
                #coherence_values.append(np.max(correlation))
             #   coherence_values.append(nccScore)
            log_DT[start:start+self.hop_size] = np.ones(self.hop_size) * (1 if double_talk else 0)
            coherence_values[start:start+self.hop_size] = np.ones(self.hop_size) * nccScore
            time_axis.append(start / self.fs)

            # Print detection status
            print(f"Time: {start / self.fs:.3f} s | Coherence: {avg_coherence:.4f} | Double Talk: {'YES' if double_talk else 'NO'}")
            if doPlots and (frame % 10 == 0):
                plt.subplot(311)
                plt.cla()
                plt.plot(block_far,label='far')
                plt.plot(block_near,label='near')
                plt.legend()
                plt.pause(.1)
                if not np.isnan(Cxy).any():
                    plt.subplot(312)
                    plt.cla()
    #                plt.plot(f,Cxy,label='Cxy')
                    plt.plot(f, Pxx, label='Pxx')
                    plt.plot(f, Pyy, label='Pyy')
                    plt.plot(f, Pxy, label='Pxy')
                    plt.title(f'frame:{frame} lag:{lag}')
                    plt.legend()
                    plt.pause(.1)
                    plt.subplot(313)
                    plt.cla()
                    #plt.plot(coherence_values, label='Pxy')
                    #plt.plot(correlation, label='correlation')
                    #plt.plot(Cxy, label='cohernce')
                    plt.plot(PxyDiff, label='PxyDiff')
                    plt.title(f'PxyDiffScore:{PxyDiffScore}')
                    plt.pause(.1)
                    plt.pause(.1)

            frame += 1
        # Plot the coherence values
        if len(coherence_values) < len(far_end):
            coherence_values = np.concatenate((coherence_values,np.zeros(len(far_end) - len(coherence_values))))
        self.plot_results(far_end,near_end,coherence_values)
        return log_DT, coherence_values

    def plot_results(self,far_end,near_end,coh):
        # Load logged results
        #times, coherences, _ = np.loadtxt(self.log_file, delimiter=',', skiprows=1, unpack=True)

        # Plot the coherence values over time
        plt.figure(figsize=(15, 6))
        plt.subplot(211)
        #plt.plot(times, coherences, color='blue', label="Coherence")
        x = [item for sublist in self.log_timestamp for item in sublist]
#        x = [row[0] for row in self.log_timestamp]
        c = [item for sublist in self.log_coh for item in sublist]
        plt.plot(x,c,self.log_coh, color='blue', label="Coherence")
        states = [item for sublist in self.log_state for item in sublist]
        #states = [row[0] for row in self.log_state]
        #cohs = [item for sublist in coh for item in sublist]
        plt.plot(x,states, color='green', label="state")
        plt.plot(coh, label="coherence")
        plt.axhline(y=self.threshold, color='red', linestyle='--', label="Threshold")
        plt.title("Block-Based Coherence Double Talk Detection (Multi-Threaded)")
        #plt.xlabel("Time (s)")
        #plt.ylabel("Coherence")
        #plt.legend()
        plt.grid()
        plt.pause(.1)
        plt.subplot(212)
        plt.plot(far_end, label="farend")
        plt.plot(near_end, label="nearend")
        #plt.plot(coh, label="coherence")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid()
        #plt.subplot(313)
        #plt.plot(coh, label="coherence")
        #plt.xlabel("Time (s)")
        #plt.legend()
        #plt.grid()
        plt.pause(.1)
        #plt.show()


# Example usage
#far_end_file = 'Hill_noisy.wav'
far_end_file = 'Fools.wav'
#near_end_file = 'echo.wav'
near_end_file = 'armstrong_noisy.wav'
Fs1, far_end = read(far_end_file)
far_end = far_end/32768.0
Fs2, near_end = read(near_end_file)
near_end = near_end/32768.0

# Ensure signals are the same length
min_len = min(len(far_end), len(near_end))
far_end = far_end[:min_len]
near_end = near_end[:min_len]
std_dev = 1e-20
noise = np.random.normal(loc=0, scale=std_dev, size=min_len)
near_end = near_end + noise
noise2 = np.random.normal(loc=0, scale=std_dev, size=min_len)
far_end = far_end + noise2

h = np.concatenate((np.zeros(50), [0,0.2,0,0,0,0,0,0,0.3,0,0,0,0,0.1,0,0,0]))
#h = np.array()
d = sig.lfilter(h,1,far_end)
mic = near_end

rng = np.random.default_rng()
x = rng.standard_normal(1000)
y = np.concatenate([rng.standard_normal(100), x])
correlation = sig.correlate(x, y, mode="full")
lags = sig.correlation_lags(x.size, y.size, mode="full")
lag = lags[np.argmax(correlation)]

dtd = BlockCoherenceDTD(fs=Fs1, block_size=1024, hop_size=512, threshold=0.6, num_threads=8, log_file="dtd_log.csv")
log_DT, coherence_values = dtd.detect(far_end, mic)

mic = (d + near_end)/2.0
dtd_echo = BlockCoherenceDTD(fs=Fs1, block_size=1024, hop_size=512, threshold=0.6, num_threads=8, log_file="dtd_log.csv")
log_DT_echo, coherence_values_echo = dtd_echo.detect(far_end, mic)
pass
