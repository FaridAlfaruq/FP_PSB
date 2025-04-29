import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.sidebar.header("ECG Filtering Parameters")

# 1) Upload file
uploaded = st.sidebar.file_uploader("Upload ECG (.txt)", type=['txt'])

if uploaded is None:
    st.sidebar.warning("Please upload a .txt ECG file.")
    st.stop()

# 2) Load text file (tab delimited, skip header)
ecg_raw = np.loadtxt(uploaded, delimiter='\t', usecols=1, skiprows=1)
ecg_copy = np.copy(ecg_raw)

# 3) Crop
min_idx = st.sidebar.number_input("Start sample index", min_value=0, value=0, step=1)
max_idx = st.sidebar.number_input("End sample index", min_value=1, value=len(ecg_raw), step=1)
ecg = ecg_raw[int(min_idx):int(max_idx)]

# 4) Zero-mean
ecg = ecg - np.mean(ecg)

# 5) User parameters
fs     = st.sidebar.number_input("Sampling freq (Hz)",     min_value=1.0, value=100.0, step=1.0)
f_low  = st.sidebar.number_input("HPF cutoff (Hz)",        min_value=0.1, value=0.5, step=0.1)
f_high = st.sidebar.number_input("LPF cutoff (Hz)",        min_value=0.1, value=15.0, step=0.1)
default_thr = 100
thr = st.sidebar.number_input("Peak detection thr (MWI)", min_value=0.0, value=float(default_thr))

# —————————————— Filter dan Pipeline ——————————————

def lpf(sig, fl, fs):
    N = len(sig); T = 1/fs; Wc = 2*np.pi*fl
    D = (4/T**2) + (2*np.sqrt(2)*Wc/T) + Wc**2
    b1 = ((8/T**2) - 2*Wc**2)/D
    b2 = ((4/T**2) - (2*np.sqrt(2)*Wc/T) + Wc**2)/D
    a0 =  Wc**2 / D; a1 = 2*Wc**2/D; a2 = a0
    y = np.zeros(N)
    for n in range(2, N):
        y[n] = (b1*y[n-1] - b2*y[n-2]
                + a0*sig[n] + a1*sig[n-1] + a2*sig[n-2])
    return y

def hpf(sig, fh, fs):
    N = len(sig); T = 1/fs; Wc = 2*np.pi*fh
    D = (4/T**2) + (2*np.sqrt(2)*Wc/T) + Wc**2
    b1 = ((8/T**2) - 2*Wc**2)/D
    b2 = ((4/T**2) - (2*np.sqrt(2)*Wc/T) + Wc**2)/D
    a0 =  (4/T**2) / D; a1 = (-8/T**2) / D; a2 = a0
    y = np.zeros(N)
    for n in range(2, N):
        y[n] = (b1*y[n-1] - b2*y[n-2]
                + a0*sig[n] + a1*sig[n-1] + a2*sig[n-2])
    return y

def derivative_5pt(sig, fs):
    N = len(sig); T = 1/fs
    y = np.zeros(N)
    for n in range(2, N-2):
        y[n] = (-sig[n-2] - 2*sig[n-1] + 2*sig[n+1] + sig[n+2]) / (8*T)
    return y

def squaring(sig):
    return sig**2

def moving_window_integration(sig, fs, win_ms=150):
    win_size = int(round(win_ms/1000 * fs))
    kernel = np.ones(win_size) / win_size
    return np.convolve(sig, kernel, mode='same')

def peaks_r(signal, threshold):
    """
    Cari indeks n yang:
      1) signal[n] > threshold
      2) signal[n] > signal[n-1]  dan signal[n] >= signal[n+1]
    """
    peaks = []
    N = len(signal)
    for n in range(1, N-1):
        if (signal[n] > threshold
            and signal[n] > signal[n-1]
            and signal[n] >= signal[n+1]):
            peaks.append(n)
    return peaks

# 6) Jalankan pipeline
sig_lpf   = lpf(ecg,       f_high, fs)
sig_bpf   = hpf(sig_lpf,   f_low,  fs)
sig_deriv = derivative_5pt(sig_bpf, fs)
sig_sq    = squaring(sig_deriv)
sig_mwi   = moving_window_integration(sig_sq, fs)

# 7) Deteksi puncak pada MWI
r_peaks = peaks_r(ecg_copy, thr)

# 8) Hitung HR
r_times      = np.array(r_peaks) / fs
rr_intervals = np.diff(r_times)
hr_inst      = 60.0 / rr_intervals if len(rr_intervals)>0 else np.array([])
hr_avg       = float(np.mean(hr_inst)) if hr_inst.size>0 else 0.0

# —————————————— Visualisasi ——————————————

st.header("Raw ECG Signal")
st.line_chart(ecg)

st.header(f"Bandpass {f_low:.1f}–{f_high:.1f} Hz")
st.line_chart(sig_bpf)

st.header("Moving Window Integration (150 ms)")
st.line_chart(sig_mwi)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(ecg_copy, label="Raw ECG")
ax.scatter(r_peaks, ecg_copy[r_peaks], c='red', marker='*', label="R-peaks")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Amplitude")
ax.legend(loc = 'lower right')
ax.set_title(f"Detected R-Peaks (avg HR: {hr_avg:.1f} bpm)")
st.pyplot(fig)

st.write(f"Average HR: **{hr_avg:.1f} bpm**")
