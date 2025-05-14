import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.header('ECG IIR Filtering With Cascade 4th Order Bandpass Filter')
upload = st.sidebar.file_uploader("Upload File.txt", type=['txt'])
st.write("Pastikan file yang Anda upload menggunakan tab ('\\t') sebagai pemisah antar kolom.")

if upload is None:
    st.sidebar.warning("Please upload a .txt or .csv file!")
    st.stop()
signal_ecg = np.loadtxt(upload, delimiter='\t', usecols=1, skiprows=1)
signal_ecg = signal_ecg - np.mean(signal_ecg)

# to get better view of the data
min_index = st.sidebar.number_input("Start sample index", min_value=0, value=0, step=1)
max_index = st.sidebar.number_input("End sample index", min_value=1, value=len(signal_ecg), step=1)

# frequency sampling
fs = st.sidebar.number_input("Sampling freq (Hz)", min_value=1.0, value=100.0, step=1.0)

# plot raw ecg
st.header("Raw ECG Signal")
fig, ax = plt.subplots()
ax.plot(signal_ecg)
ax.set_title("Sinyal ECG")
ax.set_xlabel("Index Data")
ax.set_ylabel("Amplitude")
ax.set_xlim(int(min_index), int(max_index))
ax.grid(True)
st.pyplot(fig)

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
    for n in range(2, N-2):
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
    for n in range(0, N-1):
        y[n] = (b1 * y[n-1]) - (b2 * y[n-2]) + (a0 * signal[n]) + (a1 * signal[n-1]) + (a2 * signal[n-2])
    return y

signal_ecg = lpf(signal_ecg, 100, fs) # prefilter

# plot prefitler ecg
st.header("Prefilter(LPF) ECG Signal With Cut Off 100 Hz")
fig, ax = plt.subplots()
ax.plot(signal_ecg)
ax.set_title("Prefilter ECG")
ax.set_xlabel("Index Data")
ax.set_ylabel("Amplitude")
ax.set_xlim(int(min_index), int(max_index))
ax.grid(True)
st.pyplot(fig)

def segmented_ecg(sig):
    col1, col2, col3 = st.columns(3)

    # Input untuk P wave
    t0p = col1.number_input("Start time of P wave (ms)", min_value=0, value=19, step=1)
    t1p = col1.number_input("End time of P wave (ms)", min_value=0, value=35, step=1)
    start_p, end_p = t0p, t1p
    p_wave = sig[start_p:end_p]
    index_p = np.arange(start_p, end_p)

    # Input untuk QRS complex
    t0qrs = col2.number_input("Start time of QRS complex (ms)", min_value=0, value=34, step=1)
    t1qrs = col2.number_input("End time of QRS complex (ms)", min_value=0, value=46, step=1)
    start_qrs, end_qrs = t0qrs, t1qrs
    qrs_wave = sig[start_qrs:end_qrs]
    index_qrs = np.arange(start_qrs, end_qrs)

    # Input untuk T wave
    t0t = col3.number_input("Start time of T wave (ms)", min_value=0, value=45, step=1)
    t1t = col3.number_input("End time of T wave (ms)", min_value=0, value=78, step=1)
    start_t, end_t = t0t, t1t
    t_wave = sig[start_t:end_t]
    index_t = np.arange(start_t, end_t)
    return index_t, index_p, index_qrs, p_wave, qrs_wave, t_wave

# segmented ecg
index_t, index_p, index_qrs, p_wave, qrs_wave, t_wave = segmented_ecg(signal_ecg)

# Plotting Segmented ECG
st.header("Segmentasi P, QRS, dan T Wave")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(index_p, p_wave, color='red', label='P wave')
ax.plot(index_qrs, qrs_wave, color='green', label='QRS complex')
ax.plot(index_t, t_wave, color='blue', label='T wave')
ax.set_title("Segmentasi P, QRS, dan T Wave")
ax.set_xlabel("Index Data")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True)

st.pyplot(fig)

def dft (signal, fs):
    N = len(signal)
    Re =np.zeros(N) 
    Im =np.zeros(N)
    Mag =np.zeros(N)

    for k in range(N):
        for n in range(N):
            omega = 2 * np.pi * k * n / N
            Re[k] += signal[n] * np.cos(omega)
            Im[k] -= signal[n] * np.sin(omega)

        Mag[k] = np.sqrt(Re[k] ** 2 + Im[k] ** 2)

    # Menghitung frekuensi (untuk spektrum positif)
    f = np.arange(0, N // 2) * fs / N

    return f, Mag[:N//2]

# Hitung DFT untuk setiap segmen
f_p, Mag_p = dft(p_wave, fs)
f_qrs, Mag_qrs = dft(qrs_wave, fs)
f_t, Mag_t = dft(t_wave, fs)

# Plot hasil DFT
st.header("DFT dari P, QRS, dan T Wave")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(f_p, Mag_p, color='red', label='P wave')
ax.plot(f_qrs, Mag_qrs, color='green', label='QRS complex')
ax.plot(f_t, Mag_t, color='blue', label='T wave')
ax.set_title("Spektrum Frekuensi dari P, QRS, dan T Wave")
ax.set_xlabel("Frekuensi (Hz)")
ax.set_ylabel("Magnitudo")
ax.legend()
ax.grid(True)
st.pyplot(fig)

def peak_magnitude(f_qrs, Mag_qrs):
    index_max = np.argmax(Mag_qrs)
    fc_low = f_qrs[index_max]

    if fc_low < 0.1:
        fc_low = 0.1
    
    mag_qrs_copy = np.copy(Mag_qrs)
    mag_qrs_copy[index_max] = -np.inf

    fc_high = f_qrs[np.argmax(mag_qrs_copy)]

    return fc_low, fc_high

fc_low, fc_high = peak_magnitude(f_qrs, Mag_qrs)
# to customize the frequency cut off
f_low = st.sidebar.number_input("HPF cutoff (Hz)", min_value=0.0, value=fc_low, step=0.1)
f_high = st.sidebar.number_input("LPF cutoff (Hz)", min_value=0.0, value=fc_high, step=0.1)

def frequency_response(signal, fs, fl, fh):
    N = len(signal)
    T = 1 / fs
    wc_lpf = 2 * np.pi * fl
    wc_hpf = 2 * np.pi * fh
    num_points = 1000
    omegas = np.linspace(0, np.pi, num_points)
    frequencies = omegas * fs / (2 * np.pi)  
    magnitude_response_bpf = np.zeros(num_points)

    for i, omega in enumerate(omegas):
        # High-pass filter (HPF) - perhitungan respons kompleks
        numR_hpf = (4 / T**2) * (1 - 2 * np.cos(omega) + np.cos(2 * omega))
        numI_hpf = (4 / T**2) * (2 * np.sin(omega) - np.sin(2 * omega))
        denumR_hpf = (
            wc_hpf**2 * (1 + 2 * np.cos(omega) + np.cos(2 * omega))
            + np.sqrt(2) * wc_hpf * (2 / T) * (1 - np.cos(2 * omega))
            + (4 / T**2) * (1 - 2 * np.cos(omega) + np.cos(2 * omega))
        )
        denumI_hpf = (
            wc_hpf**2 * (2 * np.sin(omega) - np.sin(2 * omega))
            + np.sqrt(2) * wc_hpf * (2 / T) * (1 - np.cos(2 * omega))
            + (4 / T**2) * (2 * np.sin(omega) - np.sin(2 * omega))
        )
        hpf_complex_response = (numR_hpf + 1j * numI_hpf) / (denumR_hpf + 1j * denumI_hpf)

        # Low-pass filter (LPF) - perhitungan respons kompleks
        numR_lpf = wc_lpf**2 * (1 + 2 * np.cos(omega) + np.cos(2 * omega))
        numI_lpf = -wc_lpf**2 * (2 * np.sin(omega) + np.sin(2 * omega))
        denumR_lpf_lpf = (
            (4 / T**2) + (2 * np.sqrt(2) * wc_lpf / T) + wc_lpf**2
            - ((8 / T**2) - 2 * wc_lpf**2) * np.cos(omega)
            + ((4 / T**2) - (2 * np.sqrt(2) * wc_lpf / T) + wc_lpf**2) * np.cos(2 * omega)
        )
        denumI_lpf_lpf = (
            ((8 / T**2) - 2 * wc_lpf**2) * np.sin(omega)
            - ((4 / T**2) - (2 * np.sqrt(2) * wc_lpf / T) + wc_lpf**2) * np.sin(2 * omega)
        )
        lpf_complex_response = (numR_lpf + 1j * numI_lpf) / (denumR_lpf_lpf + 1j * denumI_lpf_lpf)

        # Band-pass filter (BPF) - perkalian respons kompleks
        bpf_complex_response = hpf_complex_response * lpf_complex_response
        magnitude_response_bpf[i] = np.abs(bpf_complex_response)

    return frequencies, magnitude_response_bpf

def derivative(sig, fs):
    N = len(sig); T = 1/fs
    y = np.zeros(N)
    for n in range(2, N-2):
        y[n] = (-sig[n-2] - 2*sig[n-1] + 2*sig[n+1] + sig[n+2]) / (8*T)
    return y

def abs(sig):
    return np.abs(sig)

def mav(signal):
    y = np.zeros(len(signal))
    win_size = round(0.15*fs)
    sum = 0

    for j in range(win_size):
      sum += signal[j]/win_size
      y[j] = sum
    
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
sig_lpf = lpf(signal_ecg, f_high, fs)
sig_hpf = hpf(sig_lpf, f_low, fs)
sig_der = derivative(sig_hpf, fs)
sig_abs = abs(sig_der)
sig_mav = mav(sig_abs)
test_f, test_respons = frequency_response(signal_ecg, fs, f_low, f_high)

# plot graphic
st.header(f"BPF Cut Off {f_low:.1f} Hz–{f_high:.1f} Hz")
fig, ax = plt.subplots()
ax.plot(sig_hpf)
ax.set_title(f"BPF Signal")
ax.set_xlabel("Index Data")
ax.set_ylabel("Amplitude")
ax.set_xlim(int(min_index), int(max_index))
ax.grid(True)
st.pyplot(fig)

st.header("Signal Derivative")
fig, ax = plt.subplots()
ax.plot(sig_der)
ax.set_title(f"Signal Derivative")
ax.set_xlabel("Index Data")
ax.set_ylabel("Amplitude")
ax.set_xlim(int(min_index), int(max_index))
ax.grid(True)
st.pyplot(fig)

st.header("Frequency Response BPF")
plt.figure(figsize=(10, 6))
plt.plot(test_f, test_respons)
plt.xlabel('Frekuensi (Hz)')
plt.ylabel('Magnitude Response')
plt.title('Cascaded Frequency Response (HPF dan LPF)')
plt.grid(True)
st.pyplot(plt)

st.header("Signal Absolute")
fig, ax = plt.subplots()
ax.plot(sig_abs)
ax.set_title("Signal Absolute")
ax.set_xlabel("Index Data")
ax.set_ylabel("Amplitude")
ax.set_xlim(int(min_index), int(max_index))
ax.grid(True)
st.pyplot(fig)

st.header("Moving Average Window")
fig, ax = plt.subplots()
ax.plot(sig_mav)
ax.set_title(f"Moving Average Window")
ax.set_xlabel("Index Data")
ax.set_ylabel("Amplitude")
ax.set_xlim(int(min_index), int(max_index))
ax.grid(True)
st.pyplot(fig)

threshold = np.max(sig_mav) * 0.05

# r peaks detection for heart beat
r_peaks = peaks_r(signal_ecg, threshold)
r_times      = np.array(r_peaks) / fs
rr_intervals = np.diff(r_times)
hr_inst      = 60.0 / rr_intervals if len(rr_intervals)>0 else np.array([])
hr_avg       = float(np.mean(hr_inst)) if hr_inst.size>0 else 0.0


fig, ax = plt.subplots()
ax.plot(signal_ecg)
ax.scatter(r_peaks, signal_ecg[r_peaks], c='red', marker='*', label="R-peaks")
ax.set_title(f"Detected R-Peaks")
ax.set_xlabel("Index Data")
ax.set_ylabel("Amplitude")
ax.set_xlim(int(min_index), int(max_index))
ax.grid(True)
st.pyplot(fig)
st.header(f"Average HR: **{hr_avg:.1f} bpm**")
