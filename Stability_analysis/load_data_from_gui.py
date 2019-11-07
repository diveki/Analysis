import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit, leastsq
from scipy.signal import hilbert
from scipy.ndimage.filters import gaussian_filter
import gc
import scipy as sc

path = 'O:\\19_LaserData\\14_LTA1\\Automatic_recordings\\data\\frames'

fname = os.listdir(path)
fname = 'phases_20191021_180232.h5'

fullname = os.path.join(path, fname)

f = h5py.File(fullname, 'r')
data_len = f['signal0'].len()

def read_chunks(f):    
    for i,item in enumerate(f):
        if i == 0:
            data = item
        elif i < 10000:
            data = np.vstack((data, item))
        else:
            break
    return data

def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c

# @timing
def fit_sin_fast(tt, yy):

    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    ff = np.fft.fftfreq(yy.shape[0], (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = np.fft.fft(yy)[:yy.shape[0]//2]
    idx = np.argmax(np.abs(Fyy))
    guess = np.array([np.sqrt(np.std(yy)* 2) , 2.*np.pi*np.abs(ff[idx]), np.angle(Fyy[idx]), np.mean(yy)])

    try:
        popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess, method='lm')
        A, w, p, c = popt
        f = w/(2.*np.pi)
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": lambda t: A * np.sin(w*t + p) + c, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    except:
        print('There is a problem in the fit')
        return None


def test_fitter(signal, order=15):
    
    x_range = np.linspace(0, len(signal), signal.shape[0])
    z = np.poly1d(np.polyfit(x_range, signal, order))

    differ = signal-z(x_range)
    res = fit_sin_fast(x_range, differ)
    return res['phase'], x_range, res["fitfunc"](x_range)


def clean_phase(ph, smoothing=5000):
    # print('Bringing phases between -pi and pi')
    while (sum(ph[ph>=np.pi]) or sum(ph[ph<-np.pi])):
        # print('I was in while loop')
        ph = ph - np.mean(ph)
        ph[ph >= np.pi] -= 2*np.pi
        ph[ph < -np.pi] += 2*np.pi
    
    for i in range(3):
        # print('Calculating smoothed phases', i)
        aa = pd.DataFrame(ph, columns=['phase']).ewm(span=smoothing).mean()
        y = ph - aa.phase.values
        # print('Correcting phases')
        ph[y >= np.pi] -= 2*np.pi
        ph[y < -np.pi] += 2*np.pi
        ph = ph - np.mean(ph)    
    
    # print('Calculating smoothed phases', i)
    aa = pd.DataFrame(ph, columns=['phase']).rolling(smoothing).mean().fillna(method='bfill')
    y = ph - aa.phase.values
    # print('Correcting phases')
    ph[y >= np.pi] -= 2*np.pi
    ph[y < -np.pi] += 2*np.pi
    ph = ph - np.mean(ph)

    return(ph)



chunks = 0
time = f['time'][()]

while chunks+10000 <= 100000:#data_len:
    print(chunks, data_len)
    conds = slice(chunks, chunks+10000)
    if chunks == 0:
        data = read_chunks(f['signal0'][conds])
    else:
        tmp = read_chunks(f['signal0'][conds])
        data = np.vstack((data, tmp))
    chunks = chunks + 10000
    

phase = []
# phase_orig = []
for i, rows in tqdm(enumerate(data)):
    ph, xx, yy = test_fitter(rows, order=1)
    # phase_orig.append(ph)
    if i > 0:
        if (np.abs(phase[i-1] - ph) > np.pi):
            if (ph >= 0):
                ph = ph - 2*np.pi
            else:
                ph = ph + 2*np.pi
    phase.append(ph)


f.close()

def smooth(y, N):
    y_padded = np.pad(y, (N//2, N//2-1), mode = 'edge')
    yout = np.convolve(y_padded, np.ones((N, ))/N, mode = 'valid') 
    return yout    


fname_extracted = 'phases_20191021_180232_phase_extracted.h5'

f=h5py.File(os.path.join(path, fname_extracted), 'r')
phase = f['phase'][()]
time = f['time'][()]
f.close()


step = 100000
sl = 0
phclean = np.array([])
switch = False


while True:
    if switch:
        break
    if sl%1000000 == 0:
        print(sl, len(phase))
    conds = slice(sl, sl+step)
    tmp = clean_phase(phase[conds])
    phclean = np.append(phclean, tmp)
    sl = sl + step
    if sl > len(phase):
        switch = True


pp = clean_phase(phase)
start = pd.Timestamp('2019-10-21T18:02:00')
end = pd.Timestamp('2019-10-22T04:42:00')
t = np.linspace(start.value, end.value, len(pp))
t = pd.to_datetime(t)

step_short = 10000
step_long = 20000
sl = 0
ph1 = pp.copy()
switch = False

xx=np.arange(step_long)

while True:
    if switch:
        break
    if sl+step_long > len(ph1):
        switch = True
        conds_long = slice(sl, len(ph1))
        xx=np.arange(len(ph1) - sl)
    else:
        conds_long = slice(sl, sl+step_long)
    if sl%1000000 == 0:
        print(sl, len(phase))
    # conds_short = slice(sl+step_long-step_short, sl+step_long)
    
    pslice = ph1[conds_long]
    z = np.polyfit(xx, pslice, 1)
    func = np.poly1d(z)
    tmp = pslice - func(xx)
    ph1[conds_long][tmp < - np.pi] += 2*np.pi
    ph1[conds_long][tmp > np.pi] -= 2*np.pi
    sl = sl + step_short
    

ph = pd.DataFrame(pp, columns=['phase'])
ph['mean_short'] = ph.phase.rolling(2000).mean().fillna(method='bfill')
ph['mean_long'] = ph.phase.rolling(200000).mean().fillna(method='bfill')
ph['pdiff'] = ph.mean_short - ph.mean_long
ph.phase[ph.pdiff > np.pi] -= 2*np.pi
ph.phase[ph.pdiff < -np.pi] += 2*np.pi
ph['demean'] = ph.phase - ph.phase.rolling(100000).mean().fillna(method='bfill')
ph.phase[ph.demean > np.pi] -= 2*np.pi
ph.phase[ph.demean < -np.pi] += 2*np.pi
ph['smooth'] = ph.phase - ph.phase.rolling(5000).mean().fillna(method='bfill')
ph=ph.set_index(t)


## LTA1_BAS_OPC
a=pd.read_csv('O:/19_LaserData/14_LTA1/LTA1_BAS_OPC/2019-10-21_BAS_OPC_LTA1.csv')
b=pd.read_csv('O:/19_LaserData/14_LTA1/LTA1_BAS_OPC/2019-10-22_BAS_OPC_LTA1.csv')
cc = pd.concat([a, b], axis=0)
cc = cc.set_index(cc.columns[0])

tt = pd.to_datetime(cc['2019-10-21 18:00':'2019-10-22 05:00'].index.values)
temp = cc['2019-10-21 18:00':'2019-10-22 05:00']['Air temperature'].values
humid = cc['2019-10-21 18:00':'2019-10-22 05:00']['Relative humidity'].values
pressure = cc['2019-10-21 18:00':'2019-10-22 05:00']['Relative air pressure(HPa)'].values



# temperature at different points
a=pd.read_csv('O:/19_LaserData/14_LTA1/LTA1_temperature_and_humidity/2019-10-21_temperature_and_humidity_LTA1.csv')
b=pd.read_csv('O:/19_LaserData/14_LTA1/LTA1_temperature_and_humidity/2019-10-22_temperature_and_humidity_LTA1.csv')
cc = pd.concat([a, b], axis=0)
cc = cc.set_index(cc.columns[0])

time_range = '2019-10-21 18:00':'2019-10-22 05:00'
tt = pd.to_datetime(cc['2019-10-21 18:00':'2019-10-22 05:00'][cc.Location == 'POI'].index.values)
temp = cc['2019-10-21 18:00':'2019-10-22 05:00'][cc.Location == 'POI']['Temperature'].values
# humid = cc['2019-10-21 18:00':'2019-10-22 05:00']['Relative humidity'].values
# pressure = cc['2019-10-21 18:00':'2019-10-22 05:00']['Relative air pressure(HPa)'].values



