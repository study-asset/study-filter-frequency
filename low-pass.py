# import soundfile as sf;
# import sounddevice as sd;
# from scipy.signal import butter, filtfilt;

# x, sr = sf.read("original_demo.wav");
# cut_off_hz = 1000;
# order = 2;

# b, a = butter(order, cut_off_hz, btype="low");
# y = filtfilt(b, a, x, axis = 0);

import io
import numpy as np
import streamlit as st
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

st.set_page_config(page_title="Low-pass Filter Audio Lab", layout="wide")
st.title("🎧 Low-pass 필터 전/후 오디오 + 주파수(FFT) 비교")

st.markdown(
    "- WAV 파일 업로드 → 컷오프(Hz) 조절 → 원본/필터링 소리 & 주파수 그래프 비교\n"
    "- *Tip:* 컷오프를 낮출수록 고음이 사라져서 더 먹먹하게 들림"
)

uploaded = st.file_uploader("WAV 파일을 업로드하세요", type=["wav"])

# ---- helpers ----
def to_mono(x: np.ndarray) -> np.ndarray:
    # x: (N,) or (N, C)
    if x.ndim == 1:
        return x
    return x.mean(axis=1)

def lowpass(sig: np.ndarray, sr: int, cutoff_hz: float, order: int) -> np.ndarray:
    # Use SOS for numerical stability
    nyq = sr / 2.0
    cutoff = max(1.0, min(cutoff_hz, nyq - 1.0))
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, sig)

def fft_spectrum(sig: np.ndarray, sr: int):
    sig = sig.astype(np.float64)
    N = len(sig)
    if N < 8:
        return np.array([]), np.array([])
    win = np.hanning(N)
    sigw = sig * win
    fft = np.fft.rfft(sigw)
    mag = np.abs(fft) / (N + 1e-12)
    freqs = np.fft.rfftfreq(N, 1 / sr)
    return freqs, mag

def wav_bytes(sig: np.ndarray, sr: int):
    # Streamlit st.audio는 bytes/BytesIO 가능
    # float32로 저장 (클리핑 방지 위해 normalize)
    sig = sig.astype(np.float32)
    m = np.max(np.abs(sig)) + 1e-12
    sig = sig / m * 0.95
    bio = io.BytesIO()
    sf.write(bio, sig, sr, format="WAV", subtype="PCM_16")
    bio.seek(0)
    return bio.read()

# ---- UI ----
if not uploaded:
    st.info("먼저 WAV 파일을 업로드해줘.")
    st.stop()

data, sr = sf.read(uploaded)
x = to_mono(data)

st.sidebar.header("필터 설정")
cutoff = st.sidebar.slider("Cutoff (Hz)", min_value=50, max_value=int(sr/2 - 1), value=min(1200, int(sr/2 - 1)), step=10)
order = st.sidebar.slider("Filter order", min_value=1, max_value=12, value=6, step=1)

# Optional: limit duration to keep FFT responsive for very long files
max_seconds = st.sidebar.slider("분석 길이(초) (너무 긴 파일이면 줄이기)", 1, 60, 10)
Nmax = int(sr * max_seconds)
x_view = x[:Nmax]

y_view = lowpass(x_view, sr, cutoff, order)

# Audio bytes
orig_audio = wav_bytes(x_view, sr)
lp_audio = wav_bytes(y_view, sr)

col1, col2 = st.columns(2)

with col1:
    st.subheader("원본 오디오")
    st.audio(orig_audio, format="audio/wav")

with col2:
    st.subheader(f"Low-pass 오디오 (cutoff={cutoff}Hz, order={order})")
    st.audio(lp_audio, format="audio/wav")

st.divider()

# ---- plots ----
freqs_o, mag_o = fft_spectrum(x_view, sr)
freqs_y, mag_y = fft_spectrum(y_view, sr)

plot_max_hz = st.slider("그래프 최대 주파수(Hz)", 500, min(20000, int(sr/2 - 1)), min(8000, int(sr/2 - 1)), step=100)

colA, colB = st.columns(2)

with colA:
    st.subheader("주파수 스펙트럼 (원본)")
    fig = plt.figure(figsize=(8, 4))
    if len(freqs_o):
        mask = freqs_o <= plot_max_hz
        plt.plot(freqs_o[mask], mag_o[mask])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("FFT Spectrum - Original")
    st.pyplot(fig)
    plt.close(fig)

with colB:
    st.subheader("주파수 스펙트럼 (Low-pass)")
    fig = plt.figure(figsize=(8, 4))
    if len(freqs_y):
        mask = freqs_y <= plot_max_hz
        plt.plot(freqs_y[mask], mag_y[mask])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"FFT Spectrum - Low-pass (cutoff={cutoff}Hz)")
    st.pyplot(fig)
    plt.close(fig)

st.divider()

st.subheader("겹쳐서 비교 (원본 vs Low-pass)")
fig = plt.figure(figsize=(10, 4))
if len(freqs_o) and len(freqs_y):
    mask = freqs_o <= plot_max_hz
    # 같은 주파수 그리드라 가정(같은 길이/샘플레이트로 계산했으니 동일)
    plt.plot(freqs_o[mask], mag_o[mask], label="Original")
    plt.plot(freqs_y[mask], mag_y[mask], label="Low-pass")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Overlay")
    plt.legend()
st.pyplot(fig)
plt.close(fig)

st.download_button(
    label="⬇️ Low-pass 결과 WAV 다운로드",
    data=lp_audio,
    file_name=f"lowpass_{cutoff}Hz_order{order}.wav",
    mime="audio/wav",
)
