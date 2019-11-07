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


path = 'O:\\19_LaserData\\14_LTA1\\Seizmograph_data\\h5df'

fname = ['LTA1DLTR18_20191018_1030_AIR_27KG-FRAME.h5', 'LTA1DLTR18_20191018_1030_AIR_27KG-FRAME_NORUBBER.h5', 
'LTA1DLTR18_20191018_1030_AIR_27KG-FRAME+15KG-BRB.h5', 'LTA1DLTR18_20191018_AIR_27KG-FRAME_NORUBB_15KG-BR.h5']



df = pd.DataFrame()

for name in fname[:1]:
    with h5py.File(os.path.join(path, name), 'r') as f:
        tmp = {'ch0': f['ch0'][()],
               'ch2': f['ch2'][()],
               'ch1': f['ch1'][()],
               'name': name}
        df = pd.concat([df, pd.DataFrame(tmp)], axis=0)


class Plane:
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self._normal_vector()
        self._constant_term()
    
    def _normal_vector(self):
        v1 = self.p1 - self.p2
        v2 = self.p3 - self.p2
        self.normal_vector = np.cross(v1, v2)
    
    def _constant_term(self):
        self.constant = np.sum(self.normal_vector * self.p2, axis=1)
    
    def get_normal_vector(self):
        return self.normal_vector
    
    def get_constant_term(self):
        return self.constant
    
    def show_plane_equation(self):
        if self.constant.shape[0] > 1:
            print(f'  {self.normal_vector[:,0]}*x \n+ {self.normal_vector[:,1]}*y\n+ {self.normal_vector[:,2]}*z\n = {self.constant}')
        else:
            print(f'{self.normal_vector[0]}*x + {self.normal_vector[1]}*y + {self.normal_vector[2]}*z = {self.constant}')

    def angle_between_planes(self, other):
        dot_prod = np.sum(self.normal_vector * other.normal_vector, axis=1)
        sqrt_norm = np.sqrt(np.linalg.norm(self.normal_vector, axis=1) * np.linalg.norm(other.normal_vector, axis=1))
        cos_alpha = dot_prod / sqrt_norm
        return(np.arccos(cos_alpha))


tmp = df[df.name==fname[0]]

# definition of seizmograph constants
g = 9.81
sampling_rate = 16384  #Hz
deltat = 1/sampling_rate
time = np.linspace(0, tmp.shape[0], tmp.shape[0])*deltat

## fft
aa = tmp.ch0.values
N = len(aa)
yf = np.fft.fft(aa)
dv = 1 / np.max(time)
freq = (np.fft.fftfreq(N)) * N * dv

yf_filtered = yf.copy()
yf_filtered[np.abs(freq) < 20] = 0
yf_filtered[np.abs(freq) > 3200] = 0
yf_filtered[np.abs(yf_filtered) < 10] = 0

aa_filtered = np.real(np.fft.ifft(yf_filtered))

aa_filtered = pd.DataFrame(aa).rolling(5000).mean().fillna(method='bfill').values

vv = sc.integrate.cumtrapz(time, aa_filtered, initial=0)

# P1=(x1, y1, z1) = (40, 7.5, z1)
# P2=(0, 0, z2) = (0, 0, z2)
# P3=(x3, y3, z3) = (0, 50, z3)
x1 = 40
y1 = 7.5
x3 = 0.
y3 = 50


# Position from the image, left seizmograph
xx = np.repeat(x1, len(tmp.ch0.values))
yy = np.repeat(y1, len(tmp.ch0.values))
p1t = np.array([xx,yy, tmp.ch0.values]).T

# Horizontal plane coefficients
xx = np.repeat(0, p1t.shape[0])
yy = np.repeat(0, p1t.shape[0])
ztemp = x1*y3 - x3*y1
zz = np.repeat(ztemp, p1t.shape[0])
ht = np.array([xx,yy, zz]).T

# Position from the image, beamsplitter seizmograph
xx = np.repeat(0, p1t.shape[0])
yy = np.repeat(0, p1t.shape[0])
p2t = np.array([xx,yy, tmp.ch1.values]).T

# Position from the image, top seizmograph
xx = np.repeat(x3, p1t.shape[0])
yy = np.repeat(y3, p1t.shape[0])
p3t = np.array([xx,yy, tmp.ch2.values]).T


## two vectors in the plane of the seizmographs
v1 = p1t - p2t
v2 = p3t - p2t
# normal vector to the plane
nv = np.cross(v1, v2)

cos_alpha = np.sum(nv * ht, axis=1) / np.sqrt(np.linalg.norm(ht, axis=1) * np.linalg.norm(nv, axis=1))



