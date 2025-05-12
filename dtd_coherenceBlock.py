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

    def detect(self, far_end, near_end):
        # Read wav files
#        far_end, fs1 = sf.read(far_end_file, dtype='float32')
#        near_end, fs2 = sf.read(near_end_file, dtype='float32')

#        if fs1 != fs2:
#            raise ValueError("Sample rates do not match.")

 #       self.fs = fs1  # Update fs in case it's different

        # Ensure signals are the same length
        min_len = min(len(far_end), len(near_end))
        far_end = far_end[:min_len]
        near_end = near_end[:min_len]

        # Clear log file
        #with open(self.log_file, mode='w', newline='') as file:
        #    writer = csv.writer(file)
        #    writer.writerow(["Time (s)", "Coherence", "Double Talk"])

        # Fill the task queue
        for start in range(0, min_len - self.block_size + 1, self.hop_size):
            block_far = far_end[start:start + self.block_size]
            block_near = near_end[start:start + self.block_size]
            start_time = start / self.fs
            self.task_queue.put((block_near, block_far, start_time))

        # Start worker threads
        threads = []
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self.worker)
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish
        self.task_queue.join()
        for thread in threads:
            thread.join()

        # Plot the coherence values
        self.plot_results(far_end,near_end)

    def plot_results(self,far_end,near_end):
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
        plt.plot(x,states, color='green', label="state")
        plt.axhline(y=self.threshold, color='red', linestyle='--', label="Threshold")
        plt.title("Block-Based Coherence Double Talk Detection (Multi-Threaded)")
        #plt.xlabel("Time (s)")
        #plt.ylabel("Coherence")
        #plt.legend()
        plt.pause(.1)
        plt.subplot(212)
        plt.plot(far_end, label="farend")
        plt.plot(near_end, label="nearend")
        plt.xlabel("Time (s)")
        plt.legend()

        plt.pause(.1)
        plt.show()


# Example usage
far_end_file = 'Hill_noisy.wav'
#far_end_file = 'Fools.wav'
near_end_file = 'echo.wav'
#near_end_file = 'armstrong_noisy.wav'
Fs1, far_end = read(far_end_file)
far_end = far_end/32768.0
Fs2, near_end = read(near_end_file)
near_end = near_end/32768.0

h = np.concatenate((np.zeros(50), [0,0.2,0,0,0,0,0,0,0.3,0,0,0,0,0.1,0,0,0]))
#h = np.array()
d = sig.lfilter(h,1,far_end)
mic = (d + near_end)/2.0

dtd = BlockCoherenceDTD(fs=Fs1, block_size=1024, hop_size=512, threshold=0.6, num_threads=8, log_file="dtd_log.csv")
dtd.detect(far_end, mic)
