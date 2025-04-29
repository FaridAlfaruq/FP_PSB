import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.header('ECG IIR Filtering With Cascade Bandpass Filter')
upload = st.sidebar.file_uploader("Upload File.txt", type=['txt'])

if upload is None:
    st.sidebar.warning("Please upload a .txt or .csv file!")
    st.stop()

signal_ecg = np.loadtxt(upload, delimiter='\t', usecols=1, skiprows=1)
sig = np.copy(signal_ecg)

# so that the web doesn't lag, so I cut the amount of data in this process
min_index = st.sidebar.number_input("Start sample index", min_value=0, value=0, step=1)
max_index = st.sidebar.number_input("End sample index", min_value=1, value=len(signal_ecg), step=1)
sig = sig[int(min_index):int(max_index)]

# normalisasi data
sig = sig - np.mean(sig)

# declare parameter
default_thr = 100
fs = st.sidebar.number_input("Sampling freq (Hz)", min_value=1.0, value=100.0, step=1.0)
f_low = st.sidebar.number_input("HPF cutoff (Hz)", min_value=0.1, value=0.5, step=0.1)
f_high = st.sidebar.number_input("LPF cutoff (Hz)", min_value=0.1, value=15.0, step=0.1)
threshold = st.sidebar.number_input("Threshold", min_value=0.0, value=float(default_thr), step=1.0)

# filter
def lpf(signal, fl, fs):
    N = len(signal)
    T = 1 / fs
    Wc = 2 * np.pi * fl

    # Koefisien
    denom = (4 / T**2) + (2 * np.sqrt(2) * Wc / T) + Wc**2
    b1 = ((8 / T**2) - (2 * Wc**2)) / denom
    b2 = ((4 / T**2) - (2 * np.sqrt(2) * Wc / T) + Wc**2) / denom
    a0 = Wc**2 / denom
    a1 = 2 * Wc**2 / denom
    a2 = a0

    y = np.zeros(N)
    for n in range(2, N):
        y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
    
    return y
  
def hpf(signal,fh,fs):
    N = len(signal)
    T = 1/fs
    Wc = 2 * np.pi * fh

    #   koefisien
    denom = (4/T**2) + (2*np.sqrt(2)*Wc/T) + Wc**2
    b1 = ((8/T**2) - 2*Wc**2)/ denom
    b2 = ((4/T**2) - (2*np.sqrt(2)*Wc/T) + Wc**2)/ denom
    a0 = (4/T**2) / denom
    a1 = (-8/T**2) / denom
    a2 = a0
    y = np.zeros(N)
    for n in range(2, N):
        y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
    return y

def derivative(sig, fs):
    N = len(sig); T = 1/fs
    y = np.zeros(N)
    for n in range(2, N-2):
        y[n] = (-sig[n-2] - 2*sig[n-1] + 2*sig[n+1] + sig[n+2]) / (8*T)
    return y

def squaring(sig):
    return sig**2

def mav(signal):
    y = np.zeros(len(signal))
    win_size = round(0.150 * fs)
    sum = 0

    # Calculate the sum for the first N terms
    for j in range(win_size):
      sum += signal[j]/win_size
      y[j] = sum
    
    # Apply the moving window integration using the equation given
    for index in range(win_size,len(signal)):  
      sum += signal[index]/win_size
      sum -= signal[index-win_size]/win_size
      y[index] = sum

    return y

def peaks_r(signal, threshold):
    y = []
    N = len(signal)
    for n in range(1, N-1):
        if (signal[n] > threshold
            and signal[n] > signal[n-1]
            and signal[n] >= signal[n+1]):
            y.append(n)
    return y

# run filtering
sig_lpf = lpf(sig, f_high, fs)
sig_hpf = hpf(sig_lpf, f_low, fs)
sig_der = derivative(sig_hpf, fs)
sig_sqr = squaring(sig_der)
sig_mav = mav(sig_sqr)

# r peaks detection for heart beat
r_peaks = peaks_r(signal_ecg, threshold) # why we use signal_ecg? because the "sig" has been cut, so we use signal_ecg instead
r_times      = np.array(r_peaks) / fs
rr_intervals = np.diff(r_times)
hr_inst      = 60.0 / rr_intervals if len(rr_intervals)>0 else np.array([])
hr_avg       = float(np.mean(hr_inst)) if hr_inst.size>0 else 0.0

# plot graphic
st.header("Raw ECG Signal")
st.line_chart(sig)

st.header(f"Bandpass {f_low:.1f}–{f_high:.1f} Hz")
st.line_chart(sig_hpf)

st.header("Derivative Signal")
st.line_chart(sig_der)

st.header("Squaring Signal")
st.line_chart(sig_sqr)

st.header("Moving Window Integration (QRS complex)")
st.line_chart(sig_mav)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(signal_ecg, label="Raw ECG")
ax.scatter(r_peaks, signal_ecg[r_peaks], c='red', marker='*', label="R-peaks")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Amplitude")
ax.legend(loc = 'lower right')
ax.set_title(f"Detected R-Peaks (avg HR: {hr_avg:.1f} bpm)")
st.pyplot(fig)

st.write(f"Average HR: **{hr_avg:.1f} bpm**")