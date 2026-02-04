#!/usr/bin/env python3
# correlation_sync_test.py
"""
Cross-correlation(shift scan)로 "싱크가 얼마나 밀렸는지" 찾는 테스트 스크립트.

사용법:
  python correlation_sync_test.py

수정해볼 곳(싱크 일부러 틀리기):
  SHIFT = 17  # 여기 숫자를 바꿔서 일부러 싱크를 틀리게 만들 수 있음

출력:
- 실제로 틀린 SHIFT 값
- 스캔으로 찾은 best shift
- best shift에서의 correlation 점수
- shift별 점수 일부(또는 전체)
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple

# =========================
# 1) 여기만 바꿔서 실험해봐
# =========================
SHIFT = 17          # B가 A보다 몇 샘플 늦게(오른쪽으로) 밀렸는지
NOISE_STD = 0.08    # B에 추가되는 노이즈 크기 (0이면 완전 동일)
DRIFT = 0.00        # B에 느린 밝기 변화(베이스라인 드리프트) 추가 (0~0.5 정도 실험)
SEED = 42           # 재현성용

# 신호 설정
N = 400             # 샘플 개수
SAMPLE_RATE = 30.0  # Hz 느낌만 내기 (PPG면 30fps 같은)


def pearson_corr(x: List[float], y: List[float]) -> float:
    """Pearson correlation: -1..1"""
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must have same non-zero length")

    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n

    num = 0.0
    dx = 0.0
    dy = 0.0
    for i in range(n):
        a = x[i] - mx
        b = y[i] - my
        num += a * b
        dx += a * a
        dy += b * b

    den = math.sqrt(dx * dy)
    # 완전 평평한 신호면 표준편차=0이라 correlation 정의가 애매해짐
    if den == 0.0:
        return 0.0
    return num / den


def make_ppg_like_signal(n: int, sr: float) -> List[float]:
    """
    "PPG 느낌"만 나는 합성 신호:
    - 기본 심박(대략 1.3Hz ~ 78bpm) + 약간의 고조파 + 약한 잡음
    """
    random.seed(SEED)
    hr_hz = 1.3
    out = []
    for i in range(n):
        t = i / sr
        base = math.sin(2 * math.pi * hr_hz * t)
        harmonic = 0.35 * math.sin(2 * math.pi * (2 * hr_hz) * t + 0.6)
        # PPG처럼 살짝 뾰족한 느낌을 주기 위해 sin을 비선형으로 변형
        shaped = math.tanh(1.6 * (base + harmonic))
        noise = random.gauss(0.0, 0.02)
        out.append(shaped + noise)
    return out


def shift_right(signal: List[float], shift: int, fill: float = 0.0) -> List[float]:
    """신호를 오른쪽으로 shift만큼 민 결과. 왼쪽은 fill로 채움."""
    if shift <= 0:
        return signal[:]
    return [fill] * shift + signal[:-shift]


def add_noise_and_drift(signal: List[float], noise_std: float, drift: float, sr: float) -> List[float]:
    """신호에 노이즈 + 느린 드리프트(베이스라인 변화) 추가."""
    out = []
    for i, v in enumerate(signal):
        t = i / sr
        baseline = drift * math.sin(2 * math.pi * 0.08 * t)  # 아주 느린 변화
        out.append(v + baseline + random.gauss(0.0, noise_std))
    return out


def correlation_at_shift(a: List[float], b: List[float], shift: int) -> float:
    """
    a(t) vs b(t+shift) 형태로 비교한다는 느낌으로:
    - shift가 양수면 b를 왼쪽으로 당겨서(=a와 정렬) 겹치는 구간만 correlation
    - 겹치는 구간 길이가 너무 짧으면 0 반환
    """
    n = len(a)
    if shift < 0:
        # 음수 shift도 지원(원하면 실험 가능)
        shift = -shift
        # a를 shift만큼 왼쪽으로 당겨 b와 맞추는 형태로 바꿔서 처리
        a_seg = a[shift:]
        b_seg = b[: n - shift]
    else:
        a_seg = a[: n - shift]
        b_seg = b[shift:]

    if len(a_seg) < 20:
        return 0.0
    return pearson_corr(a_seg, b_seg)


def scan_best_shift(a: List[float], b: List[float], max_shift: int) -> Tuple[int, float, List[Tuple[int, float]]]:
    """0..max_shift 범위로 shift를 스캔해서 correlation이 최대인 shift를 찾는다."""
    scores = []
    best_shift = 0
    best_score = -999.0

    for s in range(0, max_shift + 1):
        c = correlation_at_shift(a, b, s)
        scores.append((s, c))
        if c > best_score:
            best_score = c
            best_shift = s

    return best_shift, best_score, scores


def main():
    # A 만들기
    a = make_ppg_like_signal(N, SAMPLE_RATE)

    # B는 A를 SHIFT만큼 늦게 만들고, 노이즈/드리프트 추가
    b = shift_right(a, SHIFT, fill=0.0)
    b = add_noise_and_drift(b, NOISE_STD, DRIFT, SAMPLE_RATE)

    # 현재(틀린) 상태에서 shift=0으로 그냥 비교하면?
    corr_no_align = pearson_corr(a, b)

    # shift 스캔해서 best shift 찾기 (대충 0..80 샘플)
    max_shift = 80
    best_shift, best_score, scores = scan_best_shift(a, b, max_shift=max_shift)

    print("\n=== Correlation Sync Test ===")
    print(f"- True SHIFT (you set): {SHIFT} samples")
    print(f"- Noise std: {NOISE_STD}, Drift: {DRIFT}")
    print(f"- Correlation (no alignment, shift=0): {corr_no_align:.4f}")
    print(f"- Scan range: 0..{max_shift} samples")
    print(f"- Best shift found: {best_shift} samples")
    print(f"- Best correlation: {best_score:.4f}")

    # best shift 근처만 보기 좋게 출력
    window = 8
    lo = max(0, best_shift - window)
    hi = min(max_shift, best_shift + window)

    print("\n--- Scores near best shift ---")
    for s, c in scores[lo:hi + 1]:
        mark = " <==" if s == best_shift else ""
        print(f"shift={s:3d}  corr={c: .4f}{mark}")

    # 전체를 보고 싶으면 아래 주석 해제
    # print("\n--- All scores ---")
    # for s, c in scores:
    #     print(f"shift={s:3d}  corr={c: .4f}")

    # shift를 시간(초)로도 표시
    delay_sec = best_shift / SAMPLE_RATE
    print(f"\nEstimated delay: {delay_sec:.4f} seconds (best_shift / sample_rate)\n")


if __name__ == "__main__":
    main()
