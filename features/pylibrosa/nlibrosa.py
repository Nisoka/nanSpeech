import librosa
import librosa.display
import matplotlib.pyplot as plt


def show_wave_logMelSpec_of(wavPath):
    # Load a wav file
    y, sr = librosa.load(wavPath, sr=None)
    # extract mel spectrogram feature
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
    # convert to log scale
    logmelspec = librosa.power_to_db(melspec)
    plt.figure()
    # plot a wavform
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(y, sr)
    plt.title(wavPath)
    # plot mel spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel spectrogram')
    plt.tight_layout() #保证图不重叠
# ---------------------
# 作者：z小白
# 来源：CSDN
# 原文：https://blog.csdn.net/zzc15806/article/details/79603994
# 版权声明：本文为博主原创文章，转载请附上博文链接！


if __name__ == '__main__':
    show_wave_logMelSpec_of('/data/sr-data/aishell/data_aishell/wav/train/S0047/BAC009S0047W0476.wav')
    show_wave_logMelSpec_of('/data/sr-data/aishell/data_aishell/wav/train/S0048/BAC009S0048W0490.wav')
    plt.show()
