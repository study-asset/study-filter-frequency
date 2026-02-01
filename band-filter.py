import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, sosfiltfilt, sosfreqz

# ---------- Signal ----------
def make_signal(fs=24000, seconds=2.0, noise=0.01, seed=0, tones=(100, 1000, 5000)):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs

    x = np.zeros_like(t, dtype=np.float32)
    # 톤을 여러 개 합성 (각 톤은 크기를 조금씩 다르게)
    for f, a in zip(tones, np.linspace(0.8, 0.3, num=max(len(tones), 1))):
        x += (a * np.sin(2 * np.pi * f * t)).astype(np.float32)

    if noise > 0:
        x += (noise * rng.standard_normal(len(t))).astype(np.float32)

    # normalize
    mx = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0
    x = (x / mx).astype(np.float32)
    return t, x

# ---------- Filter ----------
def butter_sos(fs, kind, cutoff, order=6):
    nyq = fs / 2
    if kind in ("low", "high"):
        wn = cutoff / nyq
        wn = float(np.clip(wn, 1e-6, 0.999999))
        sos = butter(order, wn, btype=kind, output="sos")
    else:  # band
        lo, hi = cutoff
        lo = float(np.clip(lo / nyq, 1e-6, 0.999999))
        hi = float(np.clip(hi / nyq, 1e-6, 0.999999))
        if lo >= hi:
            # 안전 처리: lo가 hi 이상이면 bandpass 불가
            lo, hi = min(lo, hi * 0.9), hi
        sos = butter(order, (lo, hi), btype="bandpass", output="sos")
    return sos

def apply_filter(x, sos):
    # 실험/비교용: filtfilt(양방향) -> 위상 지연이 거의 없음
    return sosfiltfilt(sos, x).astype(np.float32)

# ---------- FFT ----------
def fft_mag_db(x, fs):
    n = len(x)
    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(n, 1 / fs)
    mag = 20 * np.log10(np.maximum(np.abs(X), 1e-12))
    return f, mag

# ---------- Plot helpers ----------
def plot_time(t, raw, y=None, label_y="", fs=24000, zoom_ms=40):
    nshow = int(fs * (zoom_ms / 1000))
    fig = plt.figure()
    plt.plot(t[:nshow], raw[:nshow], label="raw")
    if y is not None:
        plt.plot(t[:nshow], y[:nshow], label=label_y)
    plt.title(f"Time domain (first {zoom_ms}ms)")
    plt.xlabel("Time (s)")
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

def plot_fft(raw, fs, y=None, label_y=""):
    f_raw, m_raw = fft_mag_db(raw, fs)
    fig = plt.figure()
    plt.plot(f_raw, m_raw, label="raw")
    if y is not None:
        f_y, m_y = fft_mag_db(y, fs)
        plt.plot(f_y, m_y, label=label_y)
    plt.title("FFT magnitude (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, fs / 2)
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

def plot_response(sos, fs, title):
    w, h = sosfreqz(sos, worN=4096, fs=fs)
    mag = 20*np.log10(np.maximum(np.abs(h), 1e-12))
    fig = plt.figure()
    plt.plot(w, mag)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.ylim(-80, 5)
    st.pyplot(fig)
    plt.close(fig)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Filter Playground", layout="wide")
st.title("Low / High / Band-pass 필터 시각화 (원본 vs 필터)")

with st.sidebar:
    st.header("Signal")
    fs = st.slider("Sample Rate (fs)", 8000, 48000, 24000, 1000)
    seconds = st.slider("Duration (sec)", 1.0, 5.0, 2.0, 0.5)
    noise = st.slider("Noise", 0.0, 0.1, 0.01, 0.005)
    seed = st.number_input("Random Seed", 0, 9999, 0, 1)

    st.subheader("Tones (Hz)")
    # Nyquist 고려해서 선택 가능한 톤 범위 안내 (단순 가이드)
    default_tones = [100, 1000, 5000] if fs >= 12000 else [100, 1000]
    tones = st.multiselect(
        "Include tones",
        options=[50, 100, 200, 500, 1000, 2000, 3000, 5000, 8000, 10000],
        default=[t for t in default_tones if t < fs/2]
    )
    tones = tuple(sorted([int(f) for f in tones if f < fs/2]))
    if len(tones) == 0:
        st.warning("톤이 0개라서 신호가 노이즈만 남을 수 있어요.")

    st.divider()
    st.header("Filter")
    enable = st.toggle("Enable filter", value=False)

    mode = st.selectbox("Type", ["lowpass", "highpass", "bandpass"], index=0)
    order = st.slider("Order", 2, 10, 6, 1)

    nyq = fs / 2

    # cutoff UI
    label_y = ""
    sos = None
    cutoff_info = ""

    if mode == "lowpass":
        cutoff = st.slider("Cutoff (Hz)", 1, int(min(10000, nyq - 1)), 800, 50)
        cutoff_info = f"cut={cutoff}Hz"
        if enable:
            sos = butter_sos(fs, "low", cutoff, order=order)
            label_y = f"lowpass({cutoff}Hz)"
    elif mode == "highpass":
        cutoff = st.slider("Cutoff (Hz)", 1, int(min(10000, nyq - 1)), 2000, 50)
        cutoff_info = f"cut={cutoff}Hz"
        if enable:
            sos = butter_sos(fs, "high", cutoff, order=order)
            label_y = f"highpass({cutoff}Hz)"
    else:
        lo = st.slider("Low cut (Hz)", 1, int(min(10000, nyq - 2)), 300, 50)
        hi = st.slider("High cut (Hz)", 2, int(min(12000, nyq - 1)), 3400, 50)
        if lo >= hi:
            st.error("bandpass는 low cut < high cut 이어야 해요.")
        cutoff_info = f"cut={lo}-{hi}Hz"
        if enable and lo < hi:
            sos = butter_sos(fs, "band", (lo, hi), order=order)
            label_y = f"bandpass({lo}-{hi}Hz)"

    show_response = st.checkbox("Show filter response (Gain dB)", value=False)
    zoom_ms = st.slider("Time zoom (ms)", 10, 200, 40, 10)

# 데이터 생성
t, x = make_signal(fs=fs, seconds=seconds, noise=noise, seed=seed, tones=tones)

# 필터 적용
y = None
if enable and sos is not None:
    y = apply_filter(x, sos)

# 화면 배치: 좌(시간) / 우(FFT)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Time domain")
    plot_time(t, x, y=y, label_y=label_y, fs=fs, zoom_ms=zoom_ms)

with col2:
    st.subheader("Frequency domain (FFT)")
    plot_fft(x, fs, y=y, label_y=label_y)

# 필터 응답(원하면)
if enable and sos is not None and show_response:
    st.subheader("Filter response")
    plot_response(sos, fs, f"Response: {mode} ({cutoff_info}, order={order})")

# =========================
# Sine wave demo (circle -> sine)
# =========================
st.divider()
st.header("사인파 직관 데모: 원 위를 도는 점의 y값 = 사인파")

with st.sidebar:
    st.subheader("Sine demo")
    show_demo = st.toggle("Show sine demo", value=True)
    demo_f = st.slider("Demo frequency (Hz)", 0.5, 10.0, 1.0, 0.5)
    demo_A = st.slider("Demo amplitude (A)", 0.2, 2.0, 1.0, 0.1)
    demo_phi_deg = st.slider("Demo phase (deg)", -180, 180, 0, 5)
    demo_window = st.slider("Demo window (sec)", 1.0, 5.0, 2.0, 0.5)
    demo_t = st.slider("Time t (sec)", 0.0, demo_window, 0.0, 0.01)
    demo_play = st.button("Play (2s)")
    demo_speed = st.slider("Play speed (fps)", 5, 60, 30, 1)

if show_demo:
    import matplotlib.pyplot as plt
    import numpy as np

    phi = np.deg2rad(demo_phi_deg)
    angle = 2*np.pi*demo_f*demo_t + phi

    # 원 위의 점 (cos, sin)
    px = np.cos(angle)
    py = np.sin(angle)

    # ----- Figure 1: unit circle + point -----
    fig1 = plt.figure()
    th = np.linspace(0, 2*np.pi, 500)
    plt.plot(np.cos(th), np.sin(th))  # unit circle
    plt.scatter([px], [py], s=60)     # moving point
    # y projection line
    plt.plot([px, px], [0, py], linestyle="--")
    plt.plot([0, px], [0, 0], linestyle="--")
    plt.axhline(0)
    plt.axvline(0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Point moving on a unit circle")
    plt.xlabel("x = cos(2πft+φ)")
    plt.ylabel("y = sin(2πft+φ)")
    st.pyplot(fig1)
    plt.close(fig1)

    # ----- Figure 2: sine wave y(t) = A*sin(2πft+φ) -----
    fig2 = plt.figure()
    tt = np.linspace(0, demo_window, int(2000 * demo_window))
    yy = demo_A * np.sin(2*np.pi*demo_f*tt + phi)

    y_now = demo_A * np.sin(2*np.pi*demo_f*demo_t + phi)

    plt.plot(tt, yy)
    plt.scatter([demo_t], [y_now], s=60)
    plt.axvline(demo_t, linestyle="--")
    plt.axhline(0)
    plt.title("Sine wave is the y-coordinate over time")
    plt.xlabel("time t (sec)")
    plt.ylabel("y(t) = A*sin(2πft+φ)")
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        "원 위의 점이 시간에 따라 돌 때, 그 점의 y좌표가 sin(2πft+φ)로 변하고, 그걸 t축에 따라 그리면 사인파가 됩니다."
    )

    # ----- Optional: Play animation (simple) -----
    if demo_play:
        placeholder = st.empty()
        t0 = time.time()
        duration = 2.0
        # 재생 중엔 demo_window 범위 안에서 반복
        while True:
            elapsed = time.time() - t0
            if elapsed > duration:
                break
            t_anim = (elapsed * demo_f) % demo_window

            angle2 = 2*np.pi*demo_f*t_anim + phi
            px2, py2 = np.cos(angle2), np.sin(angle2)
            y2 = demo_A * np.sin(2*np.pi*demo_f*t_anim + phi)

            with placeholder.container():
                # circle
                figA = plt.figure()
                th = np.linspace(0, 2*np.pi, 500)
                plt.plot(np.cos(th), np.sin(th))
                plt.scatter([px2], [py2], s=60)
                plt.plot([px2, px2], [0, py2], linestyle="--")
                plt.plot([0,]())

st.markdown(
"""
### 보는 포인트
- **FFT(raw)**에서 선택한 톤(예: 100/1000/5000Hz)의 **봉우리(피크)**가 보여야 정상이에요.
- 필터를 켜면(Enable filter):
  - **lowpass**: cutoff **위쪽 성분**이 내려갑니다.
  - **highpass**: cutoff **아래쪽 성분**이 내려갑니다.
  - **bandpass**: (low cut ~ high cut) **구간만 남고 양쪽이 내려갑니다.**

### 자주 헷갈리는 것
- cutoff는 “딱 잘리는 선”이라기보다 **경계(-3dB 근처)**이고,
  얼마나 급히 떨어지는지는 **order**가 결정해요.
- sampleRate(fs)가 낮으면 Nyquist(fs/2) 위 주파수는 제대로 표현이 안 돼요.
"""
)
