import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

fs = 31250.0        #Samplings frekvens
cutoff_lpass = 3        #Nedre knekkfrekvens
cutoff_hpass = 500      #Øvre knekkfrekvens
ch1 = 3
ch2 = 4
remove_samples = 10000   #Antall samples som fjernes fra begynnelsen

def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data

def butter_coeff(fs, cutoff, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, fs = fs, cutoff =cutoff_hpass ,order=6):
    b, a = butter_coeff(fs, cutoff, order)
    return signal.lfilter(b, a, data)

def butter_bandpass(sig, fs = fs, order = 6, lowcut=cutoff_lpass, highcut=cutoff_hpass ):
    # Defining sos for butterworth bandpassfilter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    sig = signal.sosfilt(sos, sig)
    return sig

def complex_ffi(data, fs, ch1=ch1, ch2=ch2, band = 0, plot_doppler = 0, plot_IF = 0):
    IFI = data[remove_samples:, ch1]
    IFQ = data[remove_samples:, ch2]

    IFI = signal.detrend(IFI, axis=0)
    IFQ = signal.detrend(IFQ, axis=0)
    if(band):
        IFI = butter_bandpass(IFI)
        IFQ = butter_bandpass(IFQ)
    else: 
        IFI = butter_lowpass_filter(IFI)
        IFQ = butter_lowpass_filter(IFQ)

    IF = IFI + 1j * IFQ

    fft = np.fft.fft(IF, n=500000)
    fft = abs(fft)

    N = len(fft)

    freqs = np.fft.fftfreq(N, 1 / fs)

    if(plot_IF):
        plt.plot(IFI)
        plt.plot(IFQ)
    if(plot_doppler):   
        plt.plot(freqs, 10*np.log10(fft))
    if(plot_doppler or plot_IF):
        plt.show()

    return fft, freqs

def dopplerSpeed(fft, freqs, f0=24.13e9, c=299792458):
    fd = freqs[np.argmax(fft)]
    vr = (fd * c) / (2 * f0)
    return vr

def plotRaw(filename, filt = 0, detrend = 0):
    sampleR, data = raspi_import(filename)
    IFI = data[remove_samples:, ch1]
    IFQ = data[remove_samples:, ch2]
    if(detrend):
        IFI = signal.detrend(IFI, axis=0)
        IFQ = signal.detrend(IFQ, axis=0)
    if(filt):
        IFI = butter_lowpass_filter(IFI)
        IFQ = butter_lowpass_filter(IFQ)
    
    fig, ax = plt.subplots()
    ax.plot(IFI)
    ax.plot(IFQ)
    ax.set(xlabel='Sample', ylabel='Amplitude') #, fontsize = 'large'
    ax.grid()
    ax.legend(['I', 'Q'], fontsize = 'large', loc='upper right')

    plt.show()

def calculate_speed_from_file(filename, band = 0):
    _, data = raspi_import(filename)
    fft, freqs = complex_ffi(data, fs, plot_doppler=0, plot_IF=0)
    speed = dopplerSpeed(fft, freqs)
    return speed

def signaltonoise(sig, axis=0, ddof=0):
    sig = np.asanyarray(sig)
    mean = sig.mean(axis)
    sd = sig.std(axis=axis, ddof=ddof)
    snr = mean/sd
    return sd, mean, snr 

def plotSpectrum(sig):
    plt.magnitude_spectrum(sig, fs)
    plt.show()

def plotDopplerSpectrum(filename, band = 0):
    _, data = raspi_import(filename)
    fft, freqs = complex_ffi(data, fs, plot_doppler=0, plot_IF=0)

    fig, ax = plt.subplots()
    plt.semilogy(freqs, fft)
    ax.set(xlabel='Frekvens[Hz]', ylabel='FFT amplitude')
    plt.show()

def plot_compare_velocity(directory):
    data_measured=[]
    data_radar=[]
    for filename in os.listdir(directory):
        measured_speed = 0
        _, data = raspi_import(os.path.join(directory, filename))
        sd1, mean1, snr1 = signaltonoise(data[remove_samples:, 3])
        sd2, mean2, snr2 = signaltonoise(data[remove_samples:, 4])
        fft, freqs = complex_ffi(data, fs, plot_IF=0, plot_doppler=0)
        speed = dopplerSpeed(fft, freqs)
        if(filename.find("fra_0.34") != -1): measured_speed = 0.34
        elif(filename.find("fra_1.33") != -1): measured_speed = 1.33
        elif(filename.find("mot_0.34") != -1): measured_speed = -0.34
        elif(filename.find("mot_1") != -1): measured_speed = -1.0  
        else: measured_speed = 0
        data_measured.append(measured_speed)
        data_radar.append(speed)
    fig, ax = plt.subplots()
    ax.scatter(data_measured, data_radar)
    line = [min(data_radar)-0.2,max(data_radar)+0.2]
    ax.plot(line, line, linestyle="dashed")

    ax.set(ylabel='Velocity from radar (m/s)', xlabel='Veleocity from stopwatch (m/s)')
    ax.grid()
    plt.show()

def print_speed_from_dir(directory, band = 0):
    f = open("data.csv", "a")
    f.write("velocity, measured velocity, SdI, SNRI, SDQ, SNRQ")
    f.write("\n")
    for filename in os.listdir(directory):
        #print(filename)
        measured_speed = 0
        _, data = raspi_import(os.path.join(directory, filename))
        sd1, mean1, snr1 = signaltonoise(data[remove_samples:, 3])
        sd2, mean2, snr2 = signaltonoise(data[remove_samples:, 4])
        fft, freqs = complex_ffi(data, fs, plot_IF=0, plot_doppler=0)
        #sd1, mean1, snr1 = signaltonoise(fft)
        speed = dopplerSpeed(fft, freqs)
        if(filename.find("fra_0.34") != -1): measured_speed = 0.34
        elif(filename.find("fra_1.33") != -1): measured_speed = 1.33
        elif(filename.find("mot_0.34") != -1): measured_speed = -0.34
        elif(filename.find("mot_1") != -1): measured_speed = -1.0  
        else: measured_speed = 0
        #print("%s, velocity: %.6f, measured velocity: %.3f sd1: %.3f, SNR1: %.3f, sd2: %.3f, SNR2: %.3f" % (filename, speed, measured_speed, sd1, 20*np.log10(snr1),  sd2, 20*np.log10(snr2)))
        
        #Print latex table
        #print("%.3f & %.3f & %.3f & %.3f & %.3f & %.3f  \\\\" % (measured_speed, speed, sd1, 20*np.log10(snr1),  sd2, 20*np.log10(snr2)))
        f.write("%.3f, %.3f, %.3f, %.3f, %.3f, %.3f" %(speed, measured_speed, sd1, 20*np.log10(snr1),  sd2, 20*np.log10(snr2)))
        f.write("\n")
    f.close()


#1) Plott rådata av (I- og Q-signalene), dopplerspektrum og beregn SNR (fra plottene).
#Plot rå data
#plotRaw("Data_bil/bil_fra_fast_1")

#Plot filtrert og detrend
#plotRaw("Data_bil/bil_fra_fast_1", 1, 1)

#plotDopplerSpectrum("Data_bil/bil_fra_fast_1")

#Plot doppler spektrum
#plotDopplerSpectrum("test/test1")
#print(calculate_speed_from_file("test/test1"))

#2) Analyse av målenøyaktighet og estimat av standardavvik (ha med plott).
#Print data fra alle målinger
#print_speed_from_dir("Radar_0904", band = 0)

#Plot measured speed vs radar speed
plot_compare_velocity("Radar_0904")