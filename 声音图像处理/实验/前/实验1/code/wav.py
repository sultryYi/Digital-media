import wave as we
import matplotlib.pyplot as plt
import numpy as np


class Wave:
    def __init__(self, filePath):
        self.audio = we.open(filePath, 'rb')
        nchannels = self.audio.getnchannels()
        sampwidth = self.audio.getsampwidth()
        self.framerate = self.audio.getframerate()
        self.nframes = self.audio.getnframes()
        comptype = self.audio.getcomptype()
        compname = self.audio.getcompname()
        self.params = self.audio.getparams()
        self.dataWav = self.audio.readframes(self.nframes)
        self.secs = self.nframes / self.framerate
        self.audio.close()

    def drawWav(self):
        dataUse = np.fromstring(self.dataWav, dtype=np.short)
        dataUse = dataUse.reshape((-1, 2))
        dataUse = dataUse.T
        time = np.arange(0, self.nframes) * (1.0 / self.framerate)
        plt.subplot(211)
        plt.plot(time[8450000:8460000],
                 dataUse[0][8450000:8460000],
                 color='green')
        plt.subplot(212)
        plt.plot(time[8450000:8460000], dataUse[1][8450000:8460000])
        plt.show()

    def getInfo(self):
        print(self.params)

    def confirm(self):
        print('  nframes / second\n= %d / %d\n= %d\n= framerate' %
              (self.nframes, self.secs, self.framerate))
        print('由于 nframes / framerate = seconds, 故1秒钟wav音频的数量与采样率有关')


filePath = 'wav/fav.wav'
wav = Wave(filePath)
wav.getInfo()
wav.drawWav()
wav.confirm()