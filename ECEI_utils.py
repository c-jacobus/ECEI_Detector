# LOAD RELEVANT LIBRARIES
import h5py
from os import path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import copy

from scipy import signal
from scipy.signal import butter, sosfiltfilt
from scipy import interpolate

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid



# NORMALIZE
def normalize(wholeArray, v, ov):
    print("normalizing each time-series...")

    v = v - np.median(ov,axis=2, keepdims=True)

    for y in range(24):
        for x in range(8):
            if(abs(np.mean(v[y,x,:])) > 0.0001):
                v[y,x,:] = v[y,x,:]/np.mean(v[y,x,:]) - 1

    print("DONE!")
    return(v)

# FREQUENCY FILTER
def frq_filter(v, low, high):
    v_filt = np.zeros((24,8,len(v[0,0,:])))

    print(f"filtering with bandpass of [{low}, {high}]...")

    sos_coeffs = butter(N=5, Wn=[low, high], btype='bandpass', output='sos')

    for y in range(24):
        print("▉", end = '')
        for x in range(8):
            v_filt[y,x,:] = sosfiltfilt(sos_coeffs, v[y,x,:])

    print(" [✓]")           
    print("DONE!")
    
    return(v_filt)

# CHANNEL MASK
def gen_mask(v, ov):
    mask = np.ones((24,8))
    
    print("screening bad pixels...")
    for y in range(24):
        print("▉", end = '')
        for x in range(8):
            siglev = np.median(v[y,x,:])
            sigstd = np.std(v[y,x,:])

            offlev = np.median(ov[y,x,:])
            offstd = np.std(ov[y,x,:])

            # check signal level
            if siglev > 0.01:
                ref = 100 * offstd / siglev
            else:
                ref = 100 

            if ref > 30:
                mask[y,x] = 0
                # print(f'LOW signal level channel ({x}, {y}), ref = {ref}%, siglevel = {siglev} V')

             # check bottom saturation
            if offstd < 0.001:
                mask[y,x] = 0
                # print(f'SAT offset data channel ({x}, {y}), offstd = {offstd}%, offlevel = {offlev} V')

            # check top saturation.               
            if sigstd < 0.01:
                mask[y,x] = 0
                # print(f'SAT signal data channel ({x}, {y}), sigstd = {sigstd}%, siglevel = {siglev} V')
                
            if sigstd > 1:
                mask[y,x] = 1
                
            if abs(np.mean(v[y,x,:])) < 1e-20:
                mask[y,x] = 1
                
            if sigstd < 1e-5:
                mask[y,x] = 1
    
    print(" [✓]")           
    print("DONE!")        
    plot_mask(mask, "Mask")
    return(mask)

# APPLY MASK (NN)
def apply_mask(tensor, mask):
    
    output =  copy.deepcopy(tensor)
    
    print("masking...")
    for y in range(24):
        print("▉", end = '')
        for x in range(8):
            if(mask[y,x] == 1):
                
                output[y,x,:] = 0
                net_influence = 0
                
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        if x+i in range(8) and y+j in range(24) and mask[y+j, x+i] == 0:
                            
                            this_influence = 1/np.sqrt(i*i + j*j)
                            net_influence += this_influence
                            
                            output[y,x,:] += this_influence * output[y+j, x+i, :]
                
                output[y,x,:] = output[y,x,:] / net_influence
    
    print(" [✓]")              
    print("DONE!")
    return(output)

# re-scaled colorbar
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

red_blue = plt.cm.RdBu_r
black_white = plt.cm.binary

# plot a slice in red_blue
def plot_frame(this_slice, name):
    thisMax = np.max(this_slice)
    thisMin = np.min(this_slice)
    mp = 1 - thisMax / (thisMax + abs(thisMin))
    this_cmap = shiftedColorMap(red_blue, midpoint=mp, name='shifted')
    plt.imshow(this_slice, cmap = this_cmap, origin = 'lower')
    plt.ylabel('Y (px)')
    plt.xlabel('X (px)')
    plt.colorbar()
    plt.title(name)
    plt.show()

# plot a mask in black_white
def plot_mask(thisMask, name):
    plt.imshow(thisMask, cmap = black_white, vmin = 0, vmax = 1, origin = 'lower')
    plt.ylabel('Y (px)')
    plt.xlabel('X (px)')
    plt.colorbar()
    plt.title(name)
    plt.show()

# cubic interpolate and contour map
def interp_contour(this_slice, name):
    x = np.arange(0, 8, 1)
    y = np.arange(0, 24, 1)
    
    this_slice = 0.05*np.tanh(this_slice/0.075)
    interp = interpolate.interp2d(x, y, this_slice, kind='cubic')

    xnew = np.arange(0, 8, 0.25)
    ynew = np.arange(0, 24, 0.25)
    interpV = interp(xnew, ynew)

    fig = plt.figure(figsize=(4, 6), dpi=80)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    cf = ax.contourf(xnew, ynew, interpV, cmap = red_blue, levels = np.arange(-0.05,0.05,0.001))
    fig.suptitle(f"{name}  |  tanh(V)", x = 0.35, y = 0.95, fontsize=10)
    fig.colorbar(cf)
    fig.show()
    
# spectrogram
def plot_spectrogram(series, name):
    dt = 2e-6
    fs = 1/dt
    f, t, Sxx = signal.spectrogram(series, fs, scaling='spectrum')
    plt.pcolormesh(t, f[:30], np.log(Sxx[:30,:]))
    plt.ylabel('log(Frequency [Hz])')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.title(name)
    plt.show()

# itterative frames for animation
def draw_frames(num_frames):
    for i in range(num_frames):
        dt = 2e-6
        filename = f"frame_{i:05d}.png"
        timestamp = t1 + i*dt
        timestamp = "{:.6f}".format(timestamp)

        x = np.arange(0, 8, 1)
        y = np.arange(0, 24, 1)

        this_slice = 0.05*np.tanh(v_filt[:,:,i]/0.075)
        interp = interpolate.interp2d(x, y, this_slice, kind='cubic')

        xnew = np.arange(0, 8, 0.25)
        ynew = np.arange(0, 24, 0.25)
        interpV = interp(xnew, ynew)

        fig = plt.figure(figsize=(4, 6), dpi=80)
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
        cf = ax.contourf(xnew, ynew, interpV, cmap = red_blue, levels = np.arange(-0.05,0.05,0.001))
        fig.suptitle(f"tanh(V)  |  t = {timestamp} sec", x = 0.35, y = 0.95, fontsize=10)
        fig.colorbar(cf)

        fig.savefig(filename)
        plt.close()
        
class Time_Attributes:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        if isinstance(self, SimpleNamespace) and isinstance(other, SimpleNamespace):
           return self.__dict__ == other.__dict__
        return NotImplemented
    
# time-fetch fxns
def frame_from_time(time, time_args):
    
    frame = (time - time_args.offset)/time_args.dt - time_args.fr1
    
    return(int(frame))

def index_from_time(time, time_args):
    index = (time - time_args.t1)/time_args.dt
    return(int(index))

# LOAD DATA FILE
def load_file(dir_path, shot, time_args):
    print("loading datafile:")
    
    
    fr1 = int((time_args.t1-time_args.offset)/time_args.dt)
    fr2 = int((time_args.t2-time_args.offset)/time_args.dt)
    
    print(f"SPECIFIED RANGE: seconds [{time_args.t1}, {time_args.t2}] , frames [{fr1}, {fr2}] ")

    datafile = h5py.File(path.join(dir_path + shot, "ECEI." + shot + ".GT.h5"),'r')

    for i in datafile.items():
        print(i)
        
    wholeArray = np.zeros((24,8,time_args.numFrames))

    print(f"loading {time_args.numFrames} 8x24 slices...")

    for item in datafile["/ECEI"].items():
        itemName = list(item)[0]
        filePath = '/ECEI/' + itemName + '/Voltage'
        thisPixel = datafile[filePath][:].astype(np.float32)*1e-4

        x = int(itemName[7:9]) -1
        y = int(itemName[9:11]) -1

        wholeArray[x,y,:] = thisPixel # load one 5,000,000-frame array into the specified position
        if(y == 0):
            print("▉", end = '')

    print(" [✓]")        
    print("DONE!")
    
    ofr1 = 2000
    ofr2 = 8000
    ov = wholeArray[:,:,ofr1:ofr2]
    v = wholeArray[:,:,fr1:fr2]

    v = normalize(wholeArray, v, ov)

    mask = gen_mask(v, ov)
    v_masked = apply_mask(v, mask)

    v_filt = frq_filter(v_masked, 0.02, 0.036)
    return(v_filt, fr1, fr2)
