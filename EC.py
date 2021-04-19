import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave
import recorder
import padasip as pa


fe_file = wave.open('farend.wav', 'rb')
canc_file = wave.open('canc.wav', 'wb')
#mic_file = wave.open('mic.wav')


RATE = 48000 # time resolution of the recording device (Hz)
CHUNK = int(.02*RATE)  # number of data points to read at a time
FILTERLENGTH = int(.016 * RATE)

p=pyaudio.PyAudio() # start the PyAudio class
print(p.get_default_input_device_info())
#InputStream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
#              frames_per_buffer=CHUNK,input_device_index = 1) #1 uses default input device

OutputStream = p.open(format=p.get_format_from_width(fe_file.getsampwidth()),
                channels=fe_file.getnchannels(),
                rate=fe_file.getframerate(),
                output=True)
#InputStream.start_stream()
OutputStream.start_stream()

fig, (ax1, ax2) = plt.subplots(2,1)
frames = 0
canc_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
canc_file.setnchannels(1)
canc_file.setframerate(fe_file.getframerate())
rec = recorder.Recorder(channels=1)
recfile = rec.open('mic.wav', 'wb')
recfile.start_recording()

lms = pa.filters.FilterLMS(n=FILTERLENGTH, mu=0.1)

farend_data = fe_file.readframes(CHUNK)
micdata = None
primerFrames = 0
doLMS = False

while len(farend_data) > 0 and micdata is None:
    OutputStream.write(farend_data)
    micdata = recfile.data_frame
    primerFrames += 1

print("%d primer frames, now receiving mic data" % (primerFrames))

dlyline = np.zeros(FILTERLENGTH)

def delaysample(x,dl):
    tmp = dl[0:-1]        # throw out last sample
    result = np.insert(tmp, 0, x)           # insert new sample at the beginning
    return result


try:
    while len(farend_data) > 0:
        # optionally do something to the outgoing samples
        #play the wavfile samples out the speaker
        OutputStream.write(farend_data)

        # read in mic samples
        micdata = recfile.data_frame

        # LMS
        if doLMS:
            filtOutput = np.zeros(CHUNK)
            error = np.zeros(CHUNK)
            for idx in range(0, CHUNK):
                x = farend_data[idx]
                dlyline = delaysample(x, dlyline)
                y = lms.predict(dlyline)
                e = y - x
                lms.adapt(e, dlyline)
                filtOutput[idx] = y
                error[idx] = e

            canc_file.writeframes(error)
        else:
            canc_file.writeframes(micdata)

        frames += CHUNK

        if False:
            d1 = ax1.plot(data)
            ax1.set_ylim(-32768, 32767)
            Xf = np.fft.fft(data)
            d2 = ax2.plot(abs(Xf[0:int(len(Xf) / 2)]))
            plt.show()
            plt.pause(.001)
            ax1.clear()
            ax2.clear()
        farend_data = fe_file.readframes(CHUNK)


except EOFError:
    print('Hello user it is EOF exception, please enter something and run me again')
except KeyboardInterrupt:
    print('quitting...')

#InputStream.stop_stream()
#InputStream.close()
OutputStream.stop_stream()
OutputStream.close()
p.terminate()
recfile.stop_recording()
fe_file.close()
canc_file.close()
recfile.close()
#mic_file.close()
