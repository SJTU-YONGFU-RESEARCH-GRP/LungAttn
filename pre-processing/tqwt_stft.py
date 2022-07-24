import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import shutil
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, firwin
from scipy import stats
import librosa.display as libdisplay
from pydub import AudioSegment
import multiprocessing as mp
import sys
import argparse
sys.path.append('../tqwt_tools-master/')
from tqwt_tools import DualQDecomposition
dq_params = {
    'q1': 4,
    'redundancy_1': 3,
    'stages_1': 30,
    'q2': 1,
    'redundancy_2': 3,
    'stages_2': 10,
    'lambda_1': 0.1, # regularization parameter
    'lambda_2': 0.1, #
    'mu': 0.1, # affect convergence speed
    'num_iterations': 100,
    'compute_cost_function': True
}


def clip_test(dir):
    """seprate trainset and testset"""

    txt_dir = dir + 'train_test.txt'

    with open(txt_dir, 'r') as f:
        name = []
        set_type = []
        for row in f.readlines():
            row = row.strip('\n')
            row = row.split('\t')

            name.append(row[0])
            set_type.append(row[1])

    for i in range(len(name)):
        if set_type[i] == 'test':
            shutil.move(dir + 'ICBHI_final_database/' + name[i] + '.wav', dir + 'testset/' + name[i] + '.wav')


def clip_cycle(dir, new_dir):
    """clip the record into breath cycle
    dir : trainset/testset record path
    new_dir:breath cycle save path
    """
    for file in os.listdir(dir):
        txt_name = '../ICBHI/' + file[:-4] + '.txt'
        time = np.loadtxt(txt_name)[:, 0:2]
        sound = AudioSegment.from_wav(dir + file)
        for i in range(time.shape[0]):
            start_time = time[i, 0] * 1000
            stop_time = time[i, 1] * 1000
            word = sound[start_time:stop_time]
            word.export(new_dir + file[:-4] + str(i) + '.wav', format="wav")

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def Normalization(x, pattern='min_max'):
    x = x.astype(float)
    max_x = max(x)
    min_x = min(x)
    if pattern == 'min_max':
        for i in range(len(x)):
            x[i] = float(x[i] - min_x) / (max_x - min_x)
    else:  # z-score
        x = stats.zscore(x)
    return x

def stft_and_save(sig, fs, dir, win_len, file):
    stft = librosa.stft(sig, n_fft=int(win_len * fs), hop_length=int(win_len/2 * fs), window='hann')
    sub_stft = stft[0:int(len(stft) * 4000 / fs), :]
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(sub_stft), ref=np.max), y_axis='log', x_axis='time',
                             sr=fs)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(dir + file[:-3] + 'png', cmap='Greys_r')
    plt.close()

def save_pic(file):
    dir_ck = save_dir + '/ck/'
    dir_wh = save_dir + '/wh/'
    dir_res = save_dir + '/res/'
    dir_ori = save_dir + '/ori/'
    if os.path.exists(dir_ck + file[:-3] + 'png'):
        return

    fs, sig = wav.read(wav_dir + '/' + file)
    sig = Normalization(sig)
    if fs > 4000:
        sig = butter_bandpass_filter(sig, 50, 2000, fs, order=3)
    stft_and_save(sig, fs, dir_ori, 0.02, file)

    if sig.size%2 !=0:
        sig = np.append(sig,[0])

    dq = DualQDecomposition(**dq_params)
    sig_high, sig_low = dq(sig)
    sig_res = sig - sig_high - sig_low

    stft_and_save(sig_low, fs, dir_ck, 0.02, file)
    stft_and_save(sig_high, fs, dir_wh, 0.08, file)
    stft_and_save(sig_res, fs, dir_res, 0.2, file)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

parser = argparse.ArgumentParser()
parser.add_argument('--ncpu', type=int, default=16)
parser.add_argument('--savedir', default="tqwt", type=str, help='save directory')
parser.add_argument('--wavedir', default="test", type=str, help='train or test')
args = parser.parse_args()
save_dir = "../analysis/"+args.savedir+"/"+args.wavedir
wav_dir = '../data/official/'+args.wavedir

if __name__ == '__main__':
    makedirs(save_dir + '/ori/')
    makedirs(save_dir + '/ck/')
    makedirs(save_dir + '/wh/')
    makedirs(save_dir + '/res/')
    pool = mp.Pool(processes=args.ncpu)
    file_list = []
    for file in os.listdir(wav_dir):
        if os.path.splitext(file)[1] == '.wav':
            file_list.append(file)
    pool.map(save_pic, file_list)
    pool.close()
    pool.join()
    print('Done!!!')
