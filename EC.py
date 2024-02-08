# real time echo cancellation test bench
# C. Aaron Hall
# Copyright 2022 by Garmin Ltd. or its subsidiaries.
'''
Far end audio is from an audio file, played out speakers
Near end audio is captured at mic,
echo cancelled audio is output/written to a wav file
'''


import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io.wavfile import read
import recorder
import padasip as pa

p=pyaudio.PyAudio() # start the PyAudio class
print(p.get_default_input_device_info())

fe_file = wave.open('farend.wav', 'rb')
FE_sampWidth = p.get_format_from_width(fe_file.getsampwidth())
fe_file.close()
FE_Fs,FE_file = read('farend.wav')
canc_file = wave.open('canc.wav', 'wb')
#mic_file = wave.open('mic.wav')


Fs = FE_Fs # time resolution of the recording device (Hz)
CHUNK = int(.02*Fs)  # number of data points to read at a time
FILTERLENGTH = 50  #int(.016 * Fs)

#InputStream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
#              frames_per_buffer=CHUNK,input_device_index = 1) #1 uses default input device

def output_cb(in_data,frame_count, time_info, status):
    tmp = micfile.data_frame
    if len(tmp) != frame_count:
        data = np.zeros(frame_count)
    else:
        data = tmp
    return (data, pyaudio.paContinue)

from play import play
play(FE_file[:FE_Fs],Fs=FE_Fs,nCh=1)

OutputStream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=Fs,
                output=True,
                stream_callback=output_cb)
#InputStream.start_stream()
OutputStream.start_stream()

fig, (ax1, ax2) = plt.subplots(2,1)
frames = 0
canc_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
canc_file.setnchannels(1)
canc_file.setframerate(fe_file.getframerate())
mic = recorder.Recorder(channels=1,frames_per_buffer=CHUNK, rate=Fs)
micfile = mic.open('mic.wav')
micfile.start_recording()

lms = pa.filters.FilterNLMS(n=FILTERLENGTH, mu=0.1, w="zeros")
t = FE_file[0:CHUNK]
farend_data = np.array(t/32678.0, dtype=float)
micdata = None
primerFrames = 0
doLMS = True
doPlots = False

def output_cb(in_data,frame_count, time_info, status):
    tmp = micfile.data_frame
    if len(tmp) != frame_count:
        data = np.zeros(frame_count)
    else:
        data = tmp
    return (data, pyaudio.paContinue)

#while len(farend_data) > 0 and micdata is None:
#    OutputStream.write(farend_data)
#    micdata = micfile.data_frame
#    primerFrames += 1
#OutputStream.write(farend_data)
OutputStream.start_stream()

print("%d primer frames, now receiving mic data" % (primerFrames))

dlyline = np.zeros(FILTERLENGTH)

def delaysample(x,dl):
    tmp = dl[0:-1]        # throw out last sample
    result = np.insert(tmp, 0, x)           # insert new sample at the beginning
    return result


try:
    #while len(farend_data) > 0:
    for i in range(0,len(FE_file),CHUNK):
        farend_data = FE_file[i:i+CHUNK]

        # optionally do something to the outgoing samples
        #play the wavfile samples out the speaker
        #OutputStream.write(farend_data)

        # read in mic samples
        mdata = micfile.data_frame
        if mdata:
            micdata = np.frombuffer(mdata, dtype=np.int16)
        elif (i%10 == 0):
            print("No mic data frame {0}".format(i))
            micdata = np.zeros(CHUNK)

        # LMS
        if doLMS:
            filtOutput = np.zeros(CHUNK)
            error = np.zeros(CHUNK)
            for idx in range(0, CHUNK):
                x = farend_data[idx]
                dlyline = delaysample(x, dlyline)
                y = lms.predict(dlyline)
                d = micdata[idx]
                e = y - d
                filtOutput[idx] = y
                error[idx] = e
                lms.adapt(d, dlyline)

            canc_file.writeframes(error)
        else:
            if mdata:
                canc_file.writeframes(micdata)

        frames += CHUNK

        if doPlots:
            d1 = ax1.plot(data)
            ax1.set_ylim(-32768, 32767)
            Xf = np.fft.fft(data)
            d2 = ax2.plot(abs(Xf[0:int(len(Xf) / 2)]))
            plt.show()
            plt.pause(.001)
            ax1.clear()
            ax2.clear()
        #farend_data = np.array(fe_file.readframes(CHUNK)/32678.0, dtype=float)


except EOFError:
    print('Hello user it is EOF exception, please enter something and run me again')
except KeyboardInterrupt:
    print('quitting...')

#InputStream.stop_stream()
#InputStream.close()
OutputStream.stop_stream()
OutputStream.close()
p.terminate()
micfile.stop_recording()
micfile.close()
fe_file.close()
canc_file.close()
pass
pass

#mic_file.close()
