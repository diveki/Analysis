import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd
from tqdm import tqdm

path = 'O:\\19_LaserData\\14_LTA1\\Automatic_recordings\\data\\frames'

fname = os.listdir(path)

def read_h5file(name):
    with h5py.File(name, 'r') as f:
        ph = np.array(f['phase'])
        ti = np.array(f['time'])
        #raw = np.array(f['signal'])
    return (ti, ph)

def normalize_time(t):
    t = t - t.min()
    tmp = np.linspace(0, np.round(t.max()), len(t))
    return(tmp)

exp = []
for i,fn in tqdm(enumerate(fname)):
    fullname = os.path.join(path, fn)
    ti, ph = read_h5file(fullname)
    df = pd.DataFrame({'time':ti, 'phase':ph}, index=normalize_time(ti))
    exp.append(df)

exp_mod=[]
for m in exp:
    tmp = m.copy()
    tmp=tmp.drop(columns=['time'])
    tmp.phase = tmp.phase - tmp.phase.mean()
    tmp['20p_std'] = tmp.phase.rolling(20, center=True).std()
    tmp['std'] = tmp.phase.std()
    exp_mod.append(tmp)
    # N = len(phase)
    # t = m.index.values
    # T = t.max() - t.min()
    # dw = 2*np.pi / T

# ########  Fourier calculation
# freq = np.fft.fftfreq(N) * N * dw
# yft = np.fft.fft(phase)
# yft_hamming = np.fft.fft(phase * np.hamming(N))

# plt.plot(np.fft.fftshift(freq)/2/np.pi, np.fft.fftshift(np.abs(yft))/np.max(np.abs(yft)))
# plt.plot(np.fft.fftshift(freq)/2/np.pi, np.fft.fftshift(np.abs(yft_hamming))/np.max(np.abs(yft_hamming)))
# # plt.plot(np.fft.fftshift(freq)/2/np.pi, np.fft.fftshift(np.angle(yft))/np.pi, '--r')
# plt.show()

# df = pd.DataFrame({'spectrum':np.fft.fftshift(np.abs(yft))}, index=np.fft.fftshift(freq)/2/np.pi)
# df['sp_smooth'] = df[['spectrum']].rolling(10, win_type='hamming', center=True).mean()
# df['roll_std'] = df[['spectrum']].rolling(10, center=True).std()
# df['filtered'] = df[['spectrum']]
# df.filtered[df['filtered'] < 200] = 0

