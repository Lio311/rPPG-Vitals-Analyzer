import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import tempfile
import os

# --- קבועים ---
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
PPG_MIN_HZ = 0.7  # 42 BPM
PPG_MAX_HZ = 3.0  # 180 BPM

# --- פונקציות עזר ל-Plotly ---

def plot_signal(signal_series, title, yaxis_title="Amplitude"):
    """ יצירת גרף אינטראקטיבי של אות באמצעות Plotly """
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
    """ יצירת גרף ספקטרום (FFT) באמצעות Plotly """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=yf, mode='lines', name='PSD'))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        template="plotly_dark"
    )
    return fig

# --- פונקציות ליבה לעיבוד ---

@st.cache_data
def extract_signal_from_video(video_path):
    """
    טוען וידאו, מזהה פנים, מחלץ אות PPG גולמי (ערוץ ירוק) מאזור המצח.
    מחזיר סדרת Pandas עם אינדקס זמנים.
    """
    if not os.path.exists(HAAR_CASCADE_PATH):
        st.error(f"שגיאה: הקובץ {HAAR_CASCADE_PATH} לא נמצא. אנא הורד אותו.")
        return None, 0

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("שגיאה בפתיחת קובץ הוידאו.")
        return None, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    raw_signal = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # המרת פריים לגווני אפור (יעיל יותר לזיהוי פנים)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if len(faces) > 0:
            # שימוש בפנים הראשונות שזוהו
            x, y, w, h = faces[0]
            
            # הגדרת אזור עניין (ROI) - המצח
            # בערך 20% עליונים של הפנים, ו-50% אמצעיים ברוחב
            forehead_y_start = y + int(h * 0.1)
            forehead_y_end = y + int(h * 0.25)
            forehead_x_start = x + int(w * 0.25)
            forehead_x_end = x + int(w * 0.75)
            
            roi = frame[forehead_y_start:forehead_y_end, forehead_x_start:forehead_x_end]
            
            if roi.size > 0:
                # חילוץ ערוץ ירוק (אינדקס 1 ב-BGR) וחישוב ממוצע
                green_channel_mean = np.mean(roi[:, :, 1])
                raw_signal.append(green_channel_mean)
                timestamps.append(current_time_sec)

    cap.release()
    
    if not raw_signal:
        st.warning("לא זוהו פנים בוידאו.")
        return None, 0

    # יצירת סדרת Pandas עם אינדקס זמנים
    signal_series = pd.Series(raw_signal, index=pd.to_timedelta(timestamps, unit='s'))
    return signal_series, fps


def process_signal(signal_series, fs):
    """
    מנקה, מבצע Detrend, ומסנן (Bandpass) את האות הגולמי.
    """
    # 1. טיפול בערכים חסרים (אם היו פריימים בלי פנים)
    # אינטרפולציה לינארית ומילוי קצוות
    signal = signal_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    # 2. הסרת מגמה (Detrending) - הסרת שינויים איטיים
    # שימוש בהסרת ממוצע פשוטה
    signal_detrended = signal - np.mean(signal)
    
    # 3. עיצוב מסנן Butterworth Bandpass
    nyq = 0.5 * fs
    low = PPG_MIN_HZ / nyq
    high = PPG_MAX_HZ / nyq
    b, a = butter(order=4, Wn=[low, high], btype='band')
    
    # 4. החלת המסנן (filtfilt למניעת הזזת פאזה)
    filtered_signal = filtfilt(b, a, signal_detrended)
    
    return pd.Series(filtered_signal, index=signal_series.index)


def analyze_frequency_domain(signal_series, fs):
    """
    מבצע ניתוח FFT ומחשב קצב לב (HR).
    """
    N = len(signal_series)
    if N == 0:
        return 0, np.array([]), np.array([])

    # חישוב FFT
    yf = fft(signal_series.values)
    yf_power = np.abs(yf[:N // 2])**2  # Power Spectral Density
    
    xf = fftfreq(N, 1 / fs)[:N // 2]
    
    # סינון תדרים לתחום הפיזיולוגי הרלוונטי
    valid_indices = (xf >= PPG_MIN_HZ) & (xf <= PPG_MAX_HZ)
    xf_valid = xf[valid_indices]
    yf_valid = yf_power[valid_indices]
    
    if len(yf_valid) == 0:
        return 0, xf, yf_power

    # מציאת התדר הדומיננטי (השיא) בתחום
    peak_index = np.argmax(yf_valid)
    hr_frequency = xf_valid[peak_index]
    
    hr_bpm = hr_frequency * 60
    
    return hr_bpm, xf, yf_power


def analyze_time_domain(signal_series, fs):
    """
    מבצע ניתוח במרחב הזמן (איתור פיקים) ומחשב HRV (RMSSD).
    """
    signal = signal_series.values
    
    # איתור פיקים (פעימות לב)
    # נדרש כוונון של 'prominence' ו-'distance'
    distance_min = int(fs * (60.0 / 180.0)) # מרחק מינימלי בין פעימות (לפי 180 BPM)
    
    peaks, _ = find_peaks(signal, prominence=np.std(signal)*0.2, distance=distance_min)
    
    if len(peaks) < 3:
        return 0, peaks  # לא מספיק פעימות לחישוב HRV

    # חישוב מרווחי IBI (Inter-Beat Intervals) בשניות
    ibi_sec = np.diff(peaks) / fs
    
    # חישוב RMSSD (Root Mean Square of Successive Differences)
    diff_ibi_sec = np.diff(ibi_sec)
    rmssd_ms = np.sqrt(np.mean(diff_ibi_sec**2)) * 1000  # המרה למילישניות
    
    return rmssd_ms, peaks

# --- ממשק Streamlit ---

st.set_page_config(layout="wide", page_title="rPPG - מנתח דופק מוידאו")
st.title("🔬 מנתח אותות פוטופלטיסמוגרפיה (rPPG) מבוסס וידאו")
st.markdown(f"**הערה:** ודא שהקובץ `{HAAR_CASCADE_PATH}` נמצא בתיקייה.")

uploaded_file = st.file_uploader("העלה קובץ וידאו (מומלץ וידאו קצר, 15-30 שניות, עם פנים מוארות היטב)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # שמירת הקובץ שהועלה לקובץ זמני כדי ש-OpenCV יוכל לקרוא אותו
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name

    try:
        with st.spinner("מעבד וידאו... זה עשוי לקחת דקה..."):
            raw_signal_series, fps = extract_signal_from_video(video_path)
        
        if raw_signal_series is not None and not raw_signal_series.empty:
            st.success("שלב 1: חילוץ האות הגולמי הסתיים!")
            
            with st.spinner("מעבד ומסנן אות..."):
                filtered_signal_series = process_signal(raw_signal_series, fps)
            st.success("שלב 2: סינון ועיבוד האות הסתיימו!")

            with st.spinner("מבצע ניתוח תדר וזמן..."):
                # ניתוח תדר
                hr_bpm, xf, yf_power = analyze_frequency_domain(filtered_signal_series, fps)
                
                # ניתוח זמן
                rmssd_ms, peaks = analyze_time_domain(filtered_signal_series, fps)
            st.success("שלב 3: הניתוח הסתיים!")

            
            # --- הצגת התוצאות ---
            st.header("📈 תוצאות ומדדים")
            
            col1, col2 = st.columns(2)
            col1.metric("❤️ קצב לב (HR) - מבוסס FFT", f"{hr_bpm:.1f} BPM")
            col2.metric("⏱️ שונות קצב לב (RMSSD)", f"{rmssd_ms:.2f} ms")

            # --- הצגת הגרפים ---
            st.header("📊 ניתוח גרפי")
            tab1, tab2, tab3 = st.tabs(["אות גולמי", "אות מסונן ופיקים", "ניתוח תדר (FFT)"])

            with tab1:
                st.subheader("אות PPG גולמי (מהערוץ הירוק)")
                st.plotly_chart(plot_signal(raw_signal_series, "אות גולמי"), use_container_width=True)

            with tab2:
                st.subheader("אות PPG מסונן (Bandpass) עם זיהוי פעימות")
                fig_filtered = plot_signal(filtered_signal_series, "אות מסונן", yaxis_title="Normalized Amplitude")
                # הוספת הפיקים כנקודות על הגרף
                fig_filtered.add_trace(go.Scatter(
                    x=filtered_signal_series.index[peaks],
                    y=filtered_signal_series.values[peaks],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name='Peaks (פעימות)'
                ))
                st.plotly_chart(fig_filtered, use_container_width=True)

            with tab3:
                st.subheader("ספקטרום הספק (FFT) של האות המסונן")
                fig_fft = plot_fft(xf, yf_power, "Power Spectral Density (PSD)")
                # הוספת קו המציין את הדופק שנמצא
                fig_fft.add_vrect(x0=hr_bpm/60 - 0.1, x1=hr_bpm/60 + 0.1, 
                                  fillcolor="green", opacity=0.25, line_width=0,
                                  annotation_text=f"HR: {hr_bpm:.1f} BPM", annotation_position="top left")
                st.plotly_chart(fig_fft, use_container_width=True)

    finally:
        # ניקוי הקובץ הזמני
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
