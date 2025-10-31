import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import tempfile
import os

# --- Constants ---
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
PPG_MIN_HZ = 0.7  # 42 BPM
PPG_MAX_HZ = 3.0  # 180 BPM

# --- Plotly Helper Functions ---

def plot_signal(signal_series, title, yaxis_title="Amplitude"):
    """Creates an interactive signal plot using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signal_series.index, y=signal_series.values, mode='lines', name='Signal'))
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title=yaxis_title,
        template="plotly_dark"
    )
    return fig

def plot_fft(xf, yf, title):
    """Creates a spectrum (FFT) plot using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=yf, mode='lines', name='PSD'))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        template="plotly_dark"
    )
    return fig

# --- Core Processing Functions ---

@st.cache_data
def extract_signal_from_video(video_path):
    """
    Loads video, detects faces, extracts raw PPG signal (green channel) from forehead.
    Returns a Pandas Series with a time index.
    """
    if not os.path.exists(HAAR_CASCADE_PATH):
        st.error(f"Error: {HAAR_CASCADE_PATH} not found. Please download it and place it in the same directory.")
        return None, 0

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None, 0

    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- START OF FIX ---
    # Validate the FPS. It must be high enough to satisfy Nyquist theorem for our filter.
    # Our max freq is 3.0 Hz (PPG_MAX_HZ), so FPS must be > 6.0.
    MIN_FPS_REQUIRED = PPG_MAX_HZ * 2 + 1  # Add 1 for safety margin, e.g., 7.0 Hz
    
    if fps < MIN_FPS_REQUIRED:
        st.error(f"Video FPS is too low ({fps:.2f} FPS). "
                 f"A minimum of {MIN_FPS_REQUIRED:.1f} FPS is required for this analysis. "
                 "Please use a different video file (e.g., 30 FPS).")
        cap.release()
        return None, 0
    # --- END OF FIX ---

    raw_signal = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale (more efficient for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if len(faces) > 0:
            # Use the first detected face
            x, y, w, h = faces[0]
            
            # Define Region of Interest (ROI) - The forehead
            # Approx. top 10-25% of the face, and middle 50% width
            forehead_y_start = y + int(h * 0.1)
            forehead_y_end = y + int(h * 0.25)
            forehead_x_start = x + int(w * 0.25)
            forehead_x_end = x + int(w * 0.75)
            
            roi = frame[forehead_y_start:forehead_y_end, forehead_x_start:forehead_x_end]
            
            if roi.size > 0:
                # Extract green channel (index 1 in BGR) and average it
                green_channel_mean = np.mean(roi[:, :, 1])
                raw_signal.append(green_channel_mean)
                timestamps.append(current_time_sec)

    cap.release()
    
    if not raw_signal:
        st.warning("No faces were detected in the video.")
        return None, 0

    # Create a Pandas Series with a time index
    signal_series = pd.Series(raw_signal, index=pd.to_timedelta(timestamps, unit='s'))
    return signal_series, fps


def process_signal(signal_series, fs):
    """
    Cleans, detrends, and bandpass filters the raw signal.
    """
    # 1. Handle missing values (if any frames had no face)
    # Linear interpolation and fill ends
    signal = signal_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    # 2. Detrending - Remove slow-moving changes
    # Simple mean subtraction
    signal_detrended = signal - np.mean(signal)
    
    # 3. Design Butterworth Bandpass filter
    nyq = 0.5 * fs
    low = PPG_MIN_HZ / nyq
    high = PPG_MAX_HZ / nyq
    b, a = butter(order=4, Wn=[low, high], btype='band')
    
    # 4. Apply filter (filtfilt for zero phase shift)
    filtered_signal = filtfilt(b, a, signal_detrended)
    
    return pd.Series(filtered_signal, index=signal_series.index)


def analyze_frequency_domain(signal_series, fs):
    """
    Performs FFT analysis and calculates Heart Rate (HR).
    """
    N = len(signal_series)
    if N == 0:
        return 0, np.array([]), np.array([])

    # Calculate FFT
    yf = fft(signal_series.values)
    yf_power = np.abs(yf[:N // 2])**2  # Power Spectral Density
    
    xf = fftfreq(N, 1 / fs)[:N // 2]
    
    # Filter frequencies to the relevant physiological range
    valid_indices = (xf >= PPG_MIN_HZ) & (xf <= PPG_MAX_HZ)
    xf_valid = xf[valid_indices]
    yf_valid = yf_power[valid_indices]
    
    if len(yf_valid) == 0:
        return 0, xf, yf_power

    # Find the dominant frequency (peak) in the range
    peak_index = np.argmax(yf_valid)
    hr_frequency = xf_valid[peak_index]
    
    hr_bpm = hr_frequency * 60
    
    return hr_bpm, xf, yf_power


def analyze_time_domain(signal_series, fs):
    """
    Performs time-domain analysis (peak detection) and calculates HRV (RMSSD).
    """
    signal = signal_series.values
    
    # Find peaks (heartbeats)
    # 'prominence' and 'distance' may need tuning
    distance_min = int(fs * (60.0 / 180.0)) # Min distance between beats (based on 180 BPM)
    
    peaks, _ = find_peaks(signal, prominence=np.std(signal)*0.2, distance=distance_min)
    
    if len(peaks) < 3:
        return 0, peaks  # Not enough beats to calculate HRV

    # Calculate IBI (Inter-Beat Intervals) in seconds
    ibi_sec = np.diff(peaks) / fs
    
    # Calculate RMSSD (Root Mean Square of Successive Differences)
    diff_ibi_sec = np.diff(ibi_sec)
    rmssd_ms = np.sqrt(np.mean(diff_ibi_sec**2)) * 1000  # Convert to milliseconds
    
    return rmssd_ms, peaks

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="rPPG Vitals Analyzer")
st.title("rPPG - Video-Based Vitals Analyzer")
st.markdown("**Important:** For best results, film your face while holding **perfectly still**. Do not move at all.")

uploaded_file = st.file_uploader("Upload a video file (Recommended: 15-30 seconds, face well-lit)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name

    try:
        with st.spinner("Processing video... this may take a moment..."):
            raw_signal_series, fps = extract_signal_from_video(video_path)
        
        if raw_signal_series is not None and not raw_signal_series.empty:
            st.success("Step 1: Raw signal extraction complete!")
            
            with st.spinner("Processing and filtering signal..."):
                filtered_signal_series = process_signal(raw_signal_series, fps)
            st.success("Step 2: Signal filtering and processing complete!")

            with st.spinner("Performing frequency and time domain analysis..."):
                # Frequency analysis
                hr_bpm, xf, yf_power = analyze_frequency_domain(filtered_signal_series, fs)
                
                # Time analysis
                rmssd_ms, peaks = analyze_time_domain(filtered_signal_series, fps)
            st.success("Step 3: Analysis complete!")

            
            # --- Display Results ---
            st.header("Results & Metrics")
            
            col1, col2 = st.columns(2)
            col1.metric("Heart Rate (HR) - FFT Based", f"{hr_bpm:.1f} BPM")
            col2.metric("Heart Rate Variability (RMSSD)", f"{rmssd_ms:.2f} ms")

            # --- Display Plots ---
            st.header("Graphical Analysis")
            tab1, tab2, tab3 = st.tabs(["Raw Signal", "Filtered Signal & Peaks", "Frequency Analysis (FFT)"])

            with tab1:
                st.subheader("Raw PPG Signal (from Green Channel)")
                st.plotly_chart(plot_signal(raw_signal_series, "Raw Signal"), use_container_width=True)

            with tab2:
                st.subheader("Filtered PPG Signal (Bandpass) with Peak Detection")
                fig_filtered = plot_signal(filtered_signal_series, "Filtered Signal", yaxis_title="Normalized Amplitude")
                # Add peaks as markers on the plot
                fig_filtered.add_trace(go.Scatter(
                    x=filtered_signal_series.index[peaks],
                    y=filtered_signal_series.values[peaks],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name='Peaks (Heartbeats)'
                ))
                st.plotly_chart(fig_filtered, use_container_width=True)

            with tab3:
                st.subheader("Power Spectrum (FFT) of Filtered Signal")
                fig_fft = plot_fft(xf, yf_power, "Power Spectral Density (PSD)")
                # Add a vertical line/rectangle indicating the found HR
                fig_fft.add_vrect(x0=hr_bpm/60 - 0.1, x1=hr_bpm/60 + 0.1, 
                                  fillcolor="green", opacity=0.25, line_width=0,
                                  annotation_text=f"HR: {hr_bpm:.1f} BPM", annotation_position="top left")
                st.plotly_chart(fig_fft, use_container_width=True)

    finally:
        # Clean up the temporary file
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
