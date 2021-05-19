import pyaudio
from enum import Enum
from scipy.fftpack import fft
from statistics import mean
import numpy as np
import struct


class audioReaderInput(Enum):
    wav = "wav"
    device = "device"


class audioReader:
    RATE = 44100
    CHUNK = 1024
    CHANNELS = 1
    BANDS = 32

    def __init__(self, inputType, nbands=16, deviceString="", ):
        self.BANDS = nbands * 2
        if inputType == audioReaderInput.wav:
            pass
            # TODO WAV INPUT
        else:
            self.p = pyaudio.PyAudio()
            # go w default or select by devstring
            deviceIndex = 0
            for i in range(self.p.get_device_count() if deviceString else 0):
                dev = self.p.get_device_info_by_index(i)
                print(dev)
                if deviceString in dev['name']:
                    print(dev['name'], ":", i)
                    deviceIndex = i
                    break

        self.stream = self.p.open(
            # TODO grab channels from device data
            input_device_index=deviceIndex,
            format=pyaudio.paInt16,
            channels=self.CHANNELS,
            rate=self.RATE,
            frames_per_buffer=self.CHUNK,
            input=True)

    def split_fftdata_lin(self, fft_data):
        n = fft_data.size // self.BANDS
        return np.asarray([mean(fft_data[i:(i + n)])
                           for i in range(0, fft_data.size, n)])

    def split_fftdata_log(self, fft_data):
        r = fft_data.size ** (1.0 / self.BANDS)
        bins = []
        lastb = 0
        for i in range(self.BANDS):
            a = max(int(r ** i), lastb + 1)
            b = max(int(r ** (i + 1)), a + 1)
            lastb = b
            bins.append(sum(fft_data[a:b]))

        return np.asarray(bins, dtype=np.float32)

    def getFrameFFT(self):
        fft_data = fft(struct.unpack(str(self.CHUNK * self.CHANNELS) + 'h',
                       self.stream.read(self.CHUNK, exception_on_overflow=False)))
        fft_data = np.abs(fft_data[:self.CHUNK]) * 2 / (256 * self.CHUNK)
        return self.split_fftdata_log(fft_data)[:self.BANDS // 2]

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

def main():
    ar = audioReader(audioReaderInput.device, "Background")
    for i in range(200):
        bands = ar.getFrameFFT()
        print(bands)


if __name__ == "__main__":
    main()
