"""
IMM 진도계 (컴퓨터용)

스마트폰 SensorServer 앱이 웹소켓으로 보내는 가속도·자이로 값을 받아,
안드로이드/웹 앱과 같은 방식으로 IMM 진도를 실시간 계산해 pyqtgraph 창에 보여준다.

  실시간:   python imm_live.py --host 192.168.0.10 --port 8080
  ADXL345:  python imm_live.py --serial COM3          (아두이노 우노 + adxl345_imm.ino)
  CSV 재생: python imm_live.py --csv quake.csv --unit gal --speed 5
  지도 공유: 위 명령 뒤에 --share imm_share.json  (IMM 관측 지도 사이트에 1초마다 올림)
  창 없이:  위 명령 뒤에 --nogui  (1초마다 콘솔에 출력)

필요한 패키지: numpy, websocket-client (SensorServer), pyserial (ADXL345), pyqtgraph + PyQt6 또는 PySide6 (창)
SensorServer 설정에서 샘플링 주기를 가장 빠르게(예: 5000 µs = 200 Hz) 맞춰 두세요.
"""
import argparse
import collections
import csv
import json
import math
import os
import queue
import sys
import threading
import time
import urllib.parse

import numpy as np

# ===================================================================== 상수
FS, W, N = 50, 2048, 4096            # 50 Hz 재샘플링, 40.96초 창, FFT 4096점
T0, K = 1.616, 4                     # JMA 필터 최대 이득 주기, 가속도-속도 전환 계수
TMIN, TMAX, PER_OCT = 0.1, 10.0, 6
NOISE_MARGIN, CAL_SEC, LOG_FROM = 1.5, 30, 2   # LOG_FROM: III
CLASSES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X+"]
PCTG = np.array([0, 0.07, 0.23, 0.76, 2.56, 6.86, 14.73, 31.66, 68.01, 146.14])
LOW = PCTG / 100 * 980.665
MMI_COLORS = ["#ffffff", "#bfccff", "#a0e6ff", "#80ffff", "#7aff93", "#ffff00",
              "#ffc800", "#ff9100", "#ff0000", "#c80000"]
NT = int(round(math.log2(TMAX / TMIN) * PER_OCT)) + 1
TS = TMIN * 2.0 ** (np.arange(NT) / PER_OCT)
FR = np.fft.rfftfreq(N, 1 / FS)       # 0 ~ 25 Hz, 2049점
LP_T = np.round(1.6 + 0.2 * np.arange(32), 1)   # 장주기 계급 주기 1.6~7.8초
LP_LIMITS = [5, 15, 50, 100]
SHARE_RANGES = [(0.2, 0.5), (0.5, 1.5), (1.5, 7.5)]   # 관측 지도에 올리는 구간
SHARE_ENDS = [0.2, 0.5, 1.5, 7.5]
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imm_live_settings.json")


# ===================================================================== 필터
def hcut(f):
    x2 = (np.asarray(f, float) / 10.0) ** 2
    return (1 + x2 * (0.694 + x2 * (0.241 + x2 * (0.0557 + x2 * (0.009664
            + x2 * (0.00134 + x2 * 0.000155)))))) ** -0.5


def jma_gain(f):
    f = np.asarray(f, float)
    out = np.zeros_like(f)
    p = f > 0
    out[p] = np.sqrt(1 / f[p]) * hcut(f[p]) * np.sqrt(1 - np.exp(-(f[p] / 0.5) ** 3))
    return out


FMAX = float(jma_gain(np.array([1 / T0]))[0])


def nfac(T):
    T = np.asarray(T, float)
    w = 1 / (1 + (T0 / T) ** K)
    return (T0 / T) ** w


def sig_weight(f, hc):
    """신호 가중치 W(f) = 1/n(1/f). 단주기는 가속도, 장주기는 속도 기준이 되게 한다."""
    T = np.minimum(1 / np.maximum(f, 1e-12), TMAX)
    w = 1 / nfac(T)
    return w * hcut(f) if hc else w


def imm_gain(T, hc):
    """IMM_T 필터(JMA 필터를 주기 T로 이동, 최대 이득 1) x 신호 가중치. rfft 영역."""
    g = np.zeros_like(FR)
    p = FR > 0
    g[p] = jma_gain(FR[p] * T / T0) / FMAX * sig_weight(FR[p], hc)
    return g


def smooth_hp(f, lo, hi):
    return np.where(f < lo, 0.0, np.where(f < hi, 0.5 - 0.5 * np.cos(np.pi * (f - lo) / (hi - lo)), 1.0))


PGA_GAIN = smooth_hp(FR, 0.05, 0.1)
JMA_GAIN = jma_gain(FR)


def sva_gains():
    """감쇠 5% 진동자의 절대속도 응답 전달함수 (지반 가속도 gal → cm/s), 주기별."""
    z = 0.05
    hp = smooth_hp(FR, 0.03, 0.06)
    out = []
    for Tn in LP_T:
        r = FR * Tn
        Ha = (1 + 2j * z * r) / (1 - r * r + 2j * z * r)
        with np.errstate(divide="ignore", invalid="ignore"):
            Hv = np.where(FR > 0, Ha / (2j * np.pi * np.maximum(FR, 1e-12)), 0)
        out.append(hp * Hv)
    return np.array(out)


SVA_GAINS = sva_gains()


def level_frac(v):
    """소수 진도 (반올림 방식: 각 등급 하한이 x.50)."""
    if not (v > 0):
        return 0.5
    idx = int(np.searchsorted(LOW, v, side="right") - 1)
    idx = max(idx, 0)
    if idx == 0:
        return 0.5 + min(v / LOW[1], 1) * 0.999
    n = len(LOW) - 1
    if idx >= n:
        return min(n + 0.5 + math.log(v / LOW[n]) / math.log(LOW[n] / LOW[n - 1]), n + 1.49)
    return idx + 0.5 + math.log(v / LOW[idx]) / math.log(LOW[idx + 1] / LOW[idx])


def class_of(frac):
    return max(0, min(len(CLASSES) - 1, int(math.floor(frac + 0.5)) - 1))


def fmt_frac(v):
    return f"{math.floor(v * 100 + 1e-9) / 100:.2f}"


def lp_class(sva):
    return sum(1 for l in LP_LIMITS if sva >= l)


def jma_label(I):
    edges = [(0.5, "0"), (1.5, "1"), (2.5, "2"), (3.5, "3"), (4.5, "4"), (5.0, "5약"),
             (5.5, "5강"), (6.0, "6약"), (6.5, "6강")]
    for e, lab in edges:
        if I < e:
            return "진도 " + lab
    return "진도 7"


def noise_at(arr, T):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if T <= TS[0]:
        return float(arr[0])
    if T >= TS[-1]:
        return float(arr[-1])
    x = math.log2(T / TMIN) * PER_OCT
    i = int(math.floor(x)); r = x - i
    return float(arr[i] * (arr[i + 1] / arr[i]) ** r)


# ---------------------------------------------------------------- 선택 주기
def parse_spec(text):
    """'1.5' -> (1.5, 1.5), '0.2-0.5' / '0.2~0.5' / '(0.2,0.5)' -> (0.2, 0.5)."""
    s = str(text)
    for ch in "()초 \t":
        s = s.replace(ch, "")
    for sep in "~,–":
        s = s.replace(sep, "-")
    parts = [p for p in s.split("-") if p]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if not nums or len(nums) > 2:
        return None
    a = min(TMAX, max(TMIN, nums[0]))
    b = min(TMAX, max(TMIN, nums[1])) if len(nums) > 1 else a
    return (min(a, b), max(a, b))


def spec_text(s):
    return f"{s[0]:g}" if s[0] == s[1] else f"{s[0]:g}-{s[1]:g}"


# ===================================================================== 건물 모델
def building_H(f, top, cur, T1, zeta_pct):
    """균일 전단 기둥 1~5차 모드: 지반 대비 현재 층 수평 절대가속도 전달함수 (복소수)."""
    z = zeta_pct / 100
    zr = min(1.0, max(0.0, (cur - 1) / top))
    H = np.ones_like(f, dtype=complex)
    for n in range(1, 6):
        m = 2 * n - 1
        r = f * T1 / m
        G = 4 * (1 if n % 2 else -1) / (m * math.pi)
        ph = math.sin(m * math.pi * zr / 2)
        H = H + G * ph * r * r / (1 - r * r + 2j * z * r)
    return H


def est_T1(top):
    """국내 벽식 아파트 기준식 0.0488·H^0.75 (층고 2.8 m 가정)."""
    return 0.0488 * (2.8 * top) ** 0.75


# ===================================================================== 처리기
class IMMProcessor:
    """센서 샘플을 받아 50 Hz로 재샘플링하고, 최근 40초 창으로 IMM 등을 계산한다."""

    NCH = 6   # 가속도 x, y, z (gal) + 자이로 x, y, z (rad/s)

    def __init__(self):
        self.hc = False
        self.tilt = False
        self.sel = [(0.3, 0.3), (1.5, 1.5), (5.0, 5.0)]
        self.bld = {"on": False, "top": 25, "cur": 20, "T1": None, "zeta": 5.0}
        self.noise = None
        self.reset_buffer()
        self.rebuild()

    # ---------- 설정
    def bld_T1(self):
        return self.bld["T1"] or est_T1(self.bld["top"])

    def bld_key(self):
        b = self.bld
        return f'{b["top"]}/{b["cur"]}/{self.bld_T1():.3f}/{b["zeta"]}' if b["on"] else "off"

    def rebuild(self):
        self.imm_gains = np.array([imm_gain(T, self.hc) for T in TS])
        self.sel_gains = [np.array([imm_gain(a, self.hc)] + ([imm_gain(b, self.hc)] if b != a else []))
                          for a, b in self.sel]
        self.end_gains = np.array([imm_gain(T, self.hc) for T in SHARE_ENDS])
        self.hinv = None
        if self.bld["on"]:
            H = building_H(FR, self.bld["top"], self.bld["cur"], self.bld_T1(), self.bld["zeta"])
            with np.errstate(divide="ignore", invalid="ignore"):
                hinv = np.where(np.abs(H) > 0, 1 / H, 0)
            mag = np.abs(hinv)
            hinv = np.where(mag > 3, hinv * 3 / np.maximum(mag, 1e-12), hinv)   # 역보정 최대 3배
            self.hinv = hinv

    def set_sel(self, i, spec):
        self.sel[i] = spec
        self.sel_gains[i] = np.array([imm_gain(spec[0], self.hc)] +
                                     ([imm_gain(spec[1], self.hc)] if spec[1] != spec[0] else []))

    # ---------- 버퍼와 재샘플링 (25 Hz 저역통과 1차 2단 + 20 ms 구간 평균)
    def reset_buffer(self):
        self.buf = np.zeros((self.NCH, W))
        self.wi = 0
        self.total = 0
        self.next_t = None
        self.last_raw_t = 0.0
        self.raw_count = 0
        self.gyro_count = 0
        self.bin_sum = np.zeros(self.NCH)
        self.bin_n = 0
        self.last_out = np.zeros(self.NCH)
        self.lp1 = np.zeros(self.NCH)
        self.lp2 = np.zeros(self.NCH)

    def push(self, t, ax, ay, az, gx=None, gy=None, gz=None):
        """t: 초, ax/ay/az: gal, gx/gy/gz: rad/s (없으면 None)."""
        self.raw_count += 1
        has_g = gx is not None
        if has_g:
            self.gyro_count += 1
        v = np.array([ax, ay, az, gx if has_g else 0.0, gy if has_g else 0.0, gz if has_g else 0.0])
        if self.next_t is None or t - self.last_raw_t > 1 or t < self.last_raw_t:
            self.next_t = t + 1 / FS
            self.bin_sum = v.copy(); self.bin_n = 1
            self.last_raw_t = t; self.last_out = v.copy()
            self.lp1 = v.copy(); self.lp2 = v.copy()
            return
        tau = 1 / (2 * math.pi * 25)
        al = (t - self.last_raw_t) / (tau + (t - self.last_raw_t))
        self.lp1 += al * (v - self.lp1)
        self.lp2 += al * (self.lp1 - self.lp2)
        self.last_raw_t = t
        while t >= self.next_t:
            out = self.bin_sum / self.bin_n if self.bin_n else self.last_out
            self.buf[:, self.wi] = out
            self.wi = (self.wi + 1) % W
            self.total += 1
            self.last_out = out.copy()
            self.bin_sum = np.zeros(self.NCH); self.bin_n = 0
            self.next_t += 1 / FS
        self.bin_sum += self.lp2
        self.bin_n += 1

    def window(self, Lw):
        idx = (self.wi - Lw + np.arange(Lw)) % W
        return self.buf[:, idx]

    # ---------- 계산
    @staticmethod
    def _detrend(a):
        L = a.shape[-1]
        k = np.arange(L)
        A = np.vstack([k, np.ones(L)]).T
        coef, *_ = np.linalg.lstsq(A, a.T, rcond=None)
        return a - (A @ coef).T

    def _apply_tilt(self, xs, gy_win, Lw):
        """자이로로 구한 기울기 φ에 대해 측정값에 φ×r0를 더해 새어 들어온 중력을 되돌린다."""
        r0 = xs.mean(axis=1)
        g = gy_win - gy_win.mean(axis=1, keepdims=True)       # 창 평균 = 바이어스로 보고 뺀다
        ph = np.cumsum(g, axis=1) / FS
        tt = np.cross(ph.T, r0).T                             # (3, Lw), gal
        tt = self._detrend(tt)
        Tf = np.fft.rfft(tt, n=N, axis=1) * smooth_hp(FR, 0.03, 0.06)
        tt = np.fft.irfft(Tf, n=N, axis=1)[:, :Lw]
        xs = xs + tt
        h = np.sqrt(tt[0] ** 2 + tt[1] ** 2)[int(Lw * 0.05):]
        deg = math.degrees(math.hypot(r0[0], r0[1]) / max(abs(r0[2]), 1))
        return xs, {"peak": float(h.max()) if len(h) else 0.0, "deg": deg}

    def analyze(self, peak_len=None, tilt_mode=None):
        Lw = min(self.total, W)
        if Lw < 5 * FS:
            return None
        win = self.window(Lw)
        xs = win[:3].copy()
        use_tilt = (self.tilt if tilt_mode is None else tilt_mode) and self.gyro_count > 0
        tilt = None
        if use_tilt:
            xs, tilt = self._apply_tilt(xs, win[3:], Lw)
        xs = self._detrend(xs)
        nt = int(Lw * 0.05)
        xs[:, :nt] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(nt) / nt)          # 앞 5% 테이퍼
        ne = min(int(round(3 * FS)), Lw // 4)
        xs[:, Lw - ne:] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(ne) / ne))[::-1]   # 끝 3초 테이퍼
        X = np.fft.rfft(xs, n=N, axis=1)
        frm = max(int(Lw * 0.05), Lw - (peak_len or Lw))

        def mags(G):   # G: (m, 2049) 실수 이득 → (m, Lw) 3성분 합성 크기
            y = np.fft.irfft(X[:, None, :] * G[None, :, :], n=N, axis=2)[:, :, :Lw]
            return np.sqrt((y ** 2).sum(axis=0))

        def measure(Xs):
            nonlocal X
            X_save, X = X, Xs
            A = mags(self.imm_gains)[:, frm:].max(axis=1)
            sel = [mags(g)[:, frm:].max(axis=1) for g in self.sel_gains]
            ends = mags(self.end_gains)[:, frm:].max(axis=1)
            pga_mag = mags(PGA_GAIN[None, :])[0]
            jm = np.sort(mags(JMA_GAIN[None, :])[0][frm:])[::-1]
            a03 = jm[min(int(round(0.3 * FS)), len(jm) - 1)]
            I = 2 * math.log10(max(a03, 1e-12)) + 0.94
            jma = math.floor(math.floor(I * 100 + 0.5) / 10) / 10
            v = np.fft.irfft(Xs[None, :2, :] * SVA_GAINS[:, None, :], n=N, axis=2)[:, :, :Lw]
            vm = np.sqrt((v ** 2).sum(axis=1))[:, frm:].max(axis=1)
            p = int(vm.argmax())
            X = X_save
            return {"A": A, "sel": sel, "ends": ends, "pga": float(pga_mag[frm:].max()), "pga_mag": pga_mag,
                    "jma": jma, "sva": float(vm[p]), "svaT": float(LP_T[p])}

        raw = measure(X)
        cor = None
        if self.hinv is not None:
            Xc = X.copy()
            Xc[0] *= self.hinv
            Xc[1] *= self.hinv
            cor = measure(Xc)
        return {"raw": raw, "cor": cor, "Lw": Lw, "tilt": tilt}

    def measure_T1(self):
        """최근 창 수평 스펙트럼의 0.15~3 Hz에서 가장 뚜렷한 봉우리 → 건물 고유주기."""
        Lw = min(self.total, W)
        if Lw < 20 * FS:
            return None
        xs = self._detrend(self.window(Lw)[:2])
        X = np.fft.rfft(xs, n=N, axis=1)
        P = (np.abs(X) ** 2).sum(axis=0)
        S = np.convolve(P, np.ones(5) / 5, mode="same")
        k0, k1 = int(math.ceil(0.15 * N / FS)), int(3 * N / FS)
        band = S[k0:k1 + 1]
        kb = k0 + int(band.argmax())
        med = float(np.median(band))
        if not (S[kb] > 4 * med) or kb in (k0, k1):
            return None
        y0, y1, y2 = S[kb - 1], S[kb], S[kb + 1]
        den = (y0 - 2 * y1 + y2)
        dk = (y0 - y2) / (2 * den) if den else 0.0
        return {"T": 1 / ((kb + dk) * FS / N), "ratio": float(S[kb] / med)}

    # ---------- 잡음과 판정
    def noise_set(self):
        if not self.noise:
            return None
        return self.noise["on"] if (self.tilt and self.noise.get("on")) else self.noise["off"]

    def cur_noise(self, key="raw"):
        ns = self.noise_set()
        if ns is None:
            return None
        if not self.bld["on"]:
            return ns[key]
        if self.noise["bldKey"] != self.bld_key():
            return None
        return ns["cor" if key == "raw" else "svaCor"]

    def calibrate(self):
        def pack(r):
            return {"raw": r["raw"]["A"].tolist(), "cor": r["cor"]["A"].tolist() if r["cor"] else None,
                    "svaRaw": r["raw"]["sva"], "svaCor": r["cor"]["sva"] if r["cor"] else None}
        r_off = self.analyze(CAL_SEC * FS, False)
        r_on = self.analyze(CAL_SEC * FS, True) if self.gyro_count > 0 else None
        if r_off:
            self.noise = {"hc": self.hc, "bldKey": self.bld_key(), "off": pack(r_off),
                          "on": pack(r_on) if r_on else None}
        return self.noise

    def share_values(self, prim, nz):
        """관측 지도용 [IMM_(0.2,0.5), IMM_(0.5,1.5), IMM_(1.5,7.5), 대표 IMM], 대표 주기. 잡음 이하는 0."""
        lv, best = self.judge(prim["A"], nz)
        lv = [v if v is not None else 0.0 for v in lv]
        ev = []
        for j, T in enumerate(SHARE_ENDS):
            n = noise_at(nz, T)
            a = prim["ends"][j]
            ev.append(0.0 if (n is not None and a < NOISE_MARGIN * n) else level_frac(a))
        out = []
        for a, b in SHARE_RANGES:
            f = max(ev[SHARE_ENDS.index(a)], ev[SHARE_ENDS.index(b)])
            for i, T in enumerate(TS):
                if a < T < b:
                    f = max(f, lv[i])
            out.append(f)
        rep = lv[best] if best >= 0 else 0.0
        out.append(rep)
        return out, (float(TS[best]) if best >= 0 and rep > 0 else 0.0)

    @staticmethod
    def judge(A, nz):
        lv = [None if (nz is not None and a < NOISE_MARGIN * nz[i]) else level_frac(a) for i, a in enumerate(A)]
        best = -1
        for i, v in enumerate(lv):
            if v is not None and (best < 0 or v > lv[best]):
                best = i
        return lv, best

    @staticmethod
    def eval_spec(spec, vals, A, nz):
        a, b = spec
        cand = [(a, vals[0], noise_at(nz, a))]
        if b != a:
            cand.append((b, vals[1], noise_at(nz, b)))
            for i, T in enumerate(TS):
                if a < T < b:
                    cand.append((T, A[i], nz[i] if nz is not None else None))
        best = None
        for T, v, n in cand:
            f = None if (n is not None and v < NOISE_MARGIN * n) else level_frac(v)
            if best is None or (f is not None and (best[2] is None or f > best[2])) or \
               (f is None and best[2] is None and v > best[1]):
                best = (T, v, f)
        return best


# ===================================================================== 입력: SensorServer
class SensorServerClient(threading.Thread):
    """SensorServer 웹소켓에서 가속도(+자이로)를 받아 큐에 (t, ax, ay, az, gx, gy, gz)로 넣는다."""

    def __init__(self, host, port, out_q, use_gyro=True):
        super().__init__(daemon=True)
        types = ["android.sensor.accelerometer"] + (["android.sensor.gyroscope"] if use_gyro else [])
        q = urllib.parse.quote(json.dumps(types, separators=(",", ":")))
        self.url = f"ws://{host}:{port}/sensors/connect?types={q}"
        self.q = out_q
        self.gyro = None
        self.status = "연결 전"
        self.stop_flag = False
        self.ws = None

    def handle(self, message):
        d = json.loads(message)
        typ = d.get("type", "android.sensor.accelerometer")
        v = d.get("values") or []
        t = d.get("timestamp", 0) / 1e9
        if typ == "android.sensor.gyroscope" and len(v) >= 3:
            self.gyro = (float(v[0]), float(v[1]), float(v[2]))
        elif typ == "android.sensor.accelerometer" and len(v) >= 3:
            g = self.gyro or (None, None, None)
            self.q.put((t, v[0] * 100, v[1] * 100, v[2] * 100) + g)

    def run(self):
        import websocket   # pip install websocket-client
        while not self.stop_flag:
            self.status = f"연결 중: {self.url}"
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=lambda ws: setattr(self, "status", "연결됨"),
                on_message=lambda ws, m: self.handle(m),
                on_error=lambda ws, e: setattr(self, "status", f"오류: {e}"),
                on_close=lambda ws, c, r: setattr(self, "status", "연결 끊김, 3초 뒤 재시도"))
            self.ws.run_forever()
            if not self.stop_flag:
                time.sleep(3)

    def stop(self):
        self.stop_flag = True
        if self.ws:
            self.ws.close()


# ===================================================================== 입력: 아두이노 + ADXL345
class SerialClient(threading.Thread):
    """adxl345_imm.ino가 보내는 'micros,합x,합y,합z,개수' 줄을 받아 큐에 넣는다.

    눈금(gal/카운트)은 처음 5초 동안 정지 상태의 중력 크기(1 g = 980.665 gal)로 자동 보정한다.
    --scale을 주면 그 값을 그대로 쓴다. 공칭값: ADXL345 약 3.83, ADXL355(±2g) 약 0.00383 gal/카운트.
    """

    def __init__(self, port, baud, out_q, scale=None):
        super().__init__(daemon=True)
        self.port, self.baud, self.q = port, baud, out_q
        self.scale = scale
        self.status = "연결 전"
        self.stop_flag = False
        self.ser = None

    def run(self):
        import serial   # pip install pyserial
        pre = []                       # 자동 눈금을 정하기 전 5초치 (t, cx, cy, cz)
        last_us, wraps = None, 0
        while not self.stop_flag:
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=1)
                self.status = f"{self.port} 연결됨" + ("" if self.scale else ", 눈금 맞추는 중 (5초간 가만히)")
                while not self.stop_flag:
                    line = self.ser.readline().decode("ascii", "ignore").strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        if "찾지 못했" in line or "not found" in line or "확인됨" in line:
                            self.status = line.lstrip("# ")
                        continue
                    parts = line.split(",")
                    if len(parts) != 5:
                        continue
                    try:
                        us, sx, sy, sz, n = (int(p) for p in parts)
                    except ValueError:
                        continue
                    if n <= 0:
                        continue
                    if last_us is not None and us < last_us - 2 ** 31:   # micros() 32비트 넘침
                        wraps += 1
                    last_us = us
                    t = (us + wraps * 2 ** 32) / 1e6
                    c = (sx / n, sy / n, sz / n)
                    if self.scale is None:
                        pre.append((t,) + c)
                        if len(pre) >= 500:
                            g = float(np.median([math.sqrt(a * a + b * b + d * d) for _, a, b, d in pre]))
                            self.scale = 980.665 / g if g > 0 else 980.665 / 256
                            self.status = f"{self.port} 연결됨, 눈금 {self.scale:.4g} gal/카운트 (자동)"
                            for tt, a, b, d in pre:
                                self.q.put((tt, a * self.scale, b * self.scale, d * self.scale, None, None, None))
                            pre = []
                        continue
                    self.q.put((t, c[0] * self.scale, c[1] * self.scale, c[2] * self.scale, None, None, None))
            except Exception as e:   # 포트가 없거나 케이블이 빠짐
                self.status = f"시리얼 오류: {e}. 3초 뒤 재시도"
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                time.sleep(3)

    def stop(self):
        self.stop_flag = True
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass


# ===================================================================== 출력: IMM 관측 지도 (Firebase)
class FirebaseShare(threading.Thread):
    """Firebase Realtime Database에 관측값을 올린다 (익명 로그인, REST).

    용량을 아끼려고 정수 계급(0 = 잡음 이하, 1~10 = I~X+)이 바뀐 값만 올리고,
    바뀐 게 없어도 1분마다 생존 신호(t)만 보낸다.

    설정 파일(JSON) 예:
      {"apiKey": "...", "databaseURL": "https://....firebasedatabase.app",
       "name": "12층 거실", "lat": 37.25, "lon": 127.07, "sensor": "ADXL355"}
    위치는 소수 둘째 자리(약 1 km)로 반올림해서 올린다.
    같은 관측소로 다시 참여할 수 있도록 로그인 정보를 imm_share_token.json에 저장한다.
    """

    TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imm_share_token.json")

    def __init__(self, cfg_path):
        super().__init__(daemon=True)
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.key = self.cfg["apiKey"]
        self.base = self.cfg["databaseURL"].rstrip("/")
        self.latest = None
        self.lock = threading.Lock()
        self.status = "공유 준비 중"
        self.stop_flag = False
        self.uid = self.id_token = self.refresh = None
        self.expiry = 0.0
        self.sent = 0
        self.BEAT = 60.0

    def _post(self, url, data, form=False):
        import urllib.request
        body = urllib.parse.urlencode(data).encode() if form else json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/x-www-form-urlencoded" if form else "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    def _put(self, path, value, method="PUT"):
        import urllib.request
        url = f"{self.base}/{path}.json?auth={urllib.parse.quote(self.id_token)}"
        req = urllib.request.Request(url, data=json.dumps(value).encode(), method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        self.sent += 1

    def _login(self):
        tok = None
        try:
            with open(self.TOKEN_FILE, encoding="utf-8") as f:
                tok = json.load(f)
        except (OSError, ValueError):
            pass
        if tok and tok.get("refresh") and tok.get("apiKey") == self.key:
            d = self._post(f"https://securetoken.googleapis.com/v1/token?key={self.key}",
                           {"grant_type": "refresh_token", "refresh_token": tok["refresh"]}, form=True)
            self.id_token, self.refresh, self.uid = d["id_token"], d["refresh_token"], d["user_id"]
            self.expiry = time.time() + int(d.get("expires_in", 3600)) - 120
        else:
            d = self._post(f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={self.key}",
                           {"returnSecureToken": True})
            self.id_token, self.refresh, self.uid = d["idToken"], d["refreshToken"], d["localId"]
            self.expiry = time.time() + int(d.get("expiresIn", 3600)) - 120
        with open(self.TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump({"apiKey": self.key, "refresh": self.refresh, "uid": self.uid}, f)

    def push(self, values, T):
        cls = [class_of(v) + 1 if v > 0 else 0 for v in values]
        with self.lock:
            self.latest = (cls, round(float(T), 2))

    def _explain(self, e, step):
        import urllib.error
        if isinstance(e, urllib.error.HTTPError):
            try:
                body = e.read().decode("utf-8", "ignore")[:160]
            except Exception:
                body = ""
            url = (e.url or "").split("?")[0]
            hint = ""
            if e.code == 404 and step != "로그인":
                hint = " → databaseURL이 Realtime Database 주소인지 확인하세요 (…firebasedatabase.app 또는 …firebaseio.com)"
            elif e.code == 401:
                hint = " → 보안 규칙(database.rules.json)이 게시됐는지 확인하세요"
            elif "ADMIN_ONLY_OPERATION" in body or "OPERATION_NOT_ALLOWED" in body:
                hint = " → Authentication에서 익명 로그인을 사용 설정하세요"
            elif "API_KEY_INVALID" in body or "API key not valid" in body:
                hint = " → apiKey 값을 확인하세요"
            return f"{step} 실패 HTTP {e.code} ({url}) {body}{hint}"
        return f"{step} 실패: {e}"

    def run(self):
        while not self.stop_flag:
            step = "로그인"
            try:
                self._login()
                step = "관측소 등록"
                c = self.cfg
                self._put(f"stations/{self.uid}/info", {
                    "name": str(c.get("name", "이름 없음"))[:24],
                    "lat": round(float(c["lat"]), 2), "lon": round(float(c["lon"]), 2),
                    "sensor": str(c.get("sensor", "ADXL355"))[:24], "noise": True,
                    "loc": "manual"})   # 설정 파일에 직접 입력한 위치 (지도에서는 기본으로 숨겨짐)
                self._put(f"stations/{self.uid}/online", True)
                step = "관측값 올리기"
                last_sent, last_beat = None, 0.0
                while not self.stop_flag:
                    if time.time() > self.expiry:
                        self._login()
                    with self.lock:
                        item = self.latest
                    if item is None:
                        time.sleep(0.1)
                        continue
                    cls, T = item
                    ts = {".sv": "timestamp"}
                    if last_sent is None:                        # 처음엔 전체를 한 번
                        self._put(f"stations/{self.uid}/live", {"c": cls, "T": T, "t": ts})
                    else:
                        upd = {f"live/c/{i}": c for i, c in enumerate(cls) if c != last_sent[i]}
                        if "live/c/3" in upd:
                            upd["live/T"] = T
                        if not upd and time.time() - last_beat < self.BEAT:
                            time.sleep(0.1)
                            continue
                        upd["live/t"] = ts
                        self._put(f"stations/{self.uid}", upd, method="PATCH")
                    last_sent, last_beat = list(cls), time.time()
                    self.status = f"관측 지도에 공유 중 (계급이 바뀔 때만, {self.sent}회 전송)"
            except Exception as e:
                self.status = self._explain(e, step) + ". 5초 뒤 재시도"
                print("[공유] " + self.status)
                time.sleep(5)

    def stop(self):
        self.stop_flag = True
        try:
            if self.id_token and self.uid:
                self._put(f"stations/{self.uid}/online", False)
        except Exception:
            pass


def make_client(args, q):
    if args.serial:
        return SerialClient(args.serial, args.baud, q, args.scale)
    return SensorServerClient(args.host, args.port, q, not args.no_gyro)


# ===================================================================== 입력: CSV
def load_csv(path, unit="gal", fs_in=100.0, layout="auto"):
    """한 줄에 [시간, X, Y, Z(, gx, gy, gz)] 또는 [X, Y, Z]. 구분자 쉼표·세미콜론·탭·공백."""
    scale = {"gal": 1.0, "ms2": 100.0, "g": 980.665}[unit]
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for sep in ",;\t":
                s = s.replace(sep, " ")
            try:
                nums = [float(p) for p in s.split()]
            except ValueError:
                continue
            if len(nums) >= 3:
                rows.append(nums)
    if len(rows) < 10:
        raise ValueError("숫자로 된 데이터 줄을 찾지 못했어요")
    use_time = layout == "txyz" or (layout == "auto" and len(rows[0]) >= 4)
    off = 1 if use_time else 0
    rows = [r for r in rows if len(r) >= off + 3]
    a = np.array([r[:off + 3] for r in rows])
    t = a[:, 0] if use_time else np.arange(len(a)) / fs_in
    if use_time and np.any(np.diff(t) <= 0):
        raise ValueError("시간 열이 증가하지 않는 줄이 있어요. --layout xyz 로 시도해 보세요")
    gyro = None
    if use_time and all(len(r) >= 7 for r in rows[:50]) and all(len(r) >= 7 for r in rows):
        gyro = np.array([r[4:7] for r in rows])
    return t - t[0], a[:, off:off + 3] * scale, gyro


# ===================================================================== 화면 없는 모드
def describe(proc, res):
    prim = res["cor"] or res["raw"]
    lv, best = proc.judge(prim["A"], proc.cur_noise())
    imm = f"{CLASSES[class_of(lv[best])]} ({fmt_frac(lv[best])}, T* {TS[best]:.2f}초)" if best >= 0 else "잡음 수준"
    sel = []
    for i, s in enumerate(proc.sel):
        T, v, f = proc.eval_spec(s, prim["sel"][i], prim["A"], proc.cur_noise())
        sel.append(f"IMM_{spec_text(s)}={'잡음' if f is None else CLASSES[class_of(f)] + ' ' + fmt_frac(f)}")
    return (f"IMM {imm} | PGA {prim['pga']:.1f} gal ({CLASSES[class_of(level_frac(prim['pga']))]})"
            f" | JMA {prim['jma']:.1f} | 장주기 계급 {lp_class(prim['sva'])} (Sva {prim['sva']:.1f}) | " + ", ".join(sel))


def run_headless(args, proc):
    share = FirebaseShare(args.share) if args.share else None   # CSV 재생은 main()에서 공유를 끈다
    if share:
        share.start()

    def do_share(res):
        if share and res:
            prim = res["cor"] or res["raw"]
            v, T = proc.share_values(prim, None if args.csv else proc.cur_noise())
            share.push(v, T)

    if args.csv:
        t, acc, gyro = load_csv(args.csv, args.unit, args.fs, args.layout)
        i, cursor, end = 0, 0.0, t[-1] + 5
        while cursor < end:
            upto = cursor + 1
            while i < len(t) and t[i] < upto:
                g = gyro[i] if gyro is not None else (None, None, None)
                proc.push(1000 + t[i], *acc[i], *g); i += 1
            if i >= len(t):
                for tt in np.arange(max(cursor, t[-1]) + 1 / FS, upto, 1 / FS):
                    proc.push(1000 + tt, *acc[-1])
            cursor = upto
            res = proc.analyze()
            do_share(res)
            if res:
                print(f"{min(cursor, t[-1]):6.0f}초  {describe(proc, res)}")
            if args.speed != "max":
                time.sleep(1 / float(args.speed))
        return
    q = queue.Queue()
    client = make_client(args, q)
    client.start()
    try:
        while True:
            time.sleep(1)
            n = 0
            while not q.empty():
                proc.push(*q.get()); n += 1
            res = proc.analyze()
            do_share(res)
            print(f"[{client.status}] {n} Hz  " + (describe(proc, res) if res else "데이터 모으는 중")
                  + (f"  [{share.status}]" if share else ""))
    except KeyboardInterrupt:
        client.stop()
        if share:
            share.stop()


# ===================================================================== 창 (pyqtgraph)
def run_gui(args, proc):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = pg.mkQApp("IMM 진도계")
    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    win = QtWidgets.QWidget()
    win.setWindowTitle("IMM 진도계 (" + ("ADXL345 " + args.serial if args.serial else "SensorServer") + ")")
    root = QtWidgets.QHBoxLayout(win)
    left = QtWidgets.QVBoxLayout()
    right = QtWidgets.QVBoxLayout()
    root.addLayout(left, 0)
    root.addLayout(right, 1)

    def label(text="", size=13, bold=False):
        w = QtWidgets.QLabel(text)
        f = w.font(); f.setPointSize(size); f.setBold(bold); w.setFont(f)
        w.setWordWrap(True)
        return w

    hero = label("–", 60, True)
    hero.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    hero.setMinimumWidth(360)
    hero.setStyleSheet("background:#ffffff; border:1px solid #ccc; border-radius:12px; padding:8px;")
    hero_sub = label("데이터를 모으는 중이에요", 12)
    left.addWidget(hero); left.addWidget(hero_sub)
    refs = label("", 12)
    left.addWidget(refs)

    # 선택 주기
    sel_box = QtWidgets.QGroupBox("선택 주기 (1.5 또는 0.2-0.5)")
    sel_l = QtWidgets.QGridLayout(sel_box)
    sel_edits, sel_out = [], []
    for i in range(3):
        e = QtWidgets.QLineEdit(spec_text(proc.sel[i])); e.setMaximumWidth(90)
        o = label("–", 12)
        sel_l.addWidget(QtWidgets.QLabel("IMM_"), i, 0); sel_l.addWidget(e, i, 1); sel_l.addWidget(o, i, 2)
        sel_edits.append(e); sel_out.append(o)

        def on_edit(i=i, e=e):
            sp = parse_spec(e.text())
            if sp:
                proc.set_sel(i, sp); save_settings(proc)
            e.setText(spec_text(proc.sel[i]))
        e.editingFinished.connect(on_edit)
    left.addWidget(sel_box)

    # 조작
    ctl = QtWidgets.QGroupBox("측정")
    cl = QtWidgets.QVBoxLayout(ctl)
    btn_row = QtWidgets.QHBoxLayout()
    cal_btn = QtWidgets.QPushButton("잡음 측정 (30초)")
    reset_btn = QtWidgets.QPushButton("기록 최대 초기화")
    rec_btn = QtWidgets.QPushButton("원시 기록 저장 시작")
    for b in (cal_btn, reset_btn, rec_btn):
        btn_row.addWidget(b)
    cl.addLayout(btn_row)
    hc_cb = QtWidgets.QCheckBox("10 Hz 이상 성분 깎기 (JMA 고역차단)"); hc_cb.setChecked(proc.hc)
    tilt_cb = QtWidgets.QCheckBox("자이로로 기울어짐 보정"); tilt_cb.setChecked(proc.tilt)
    cl.addWidget(hc_cb); cl.addWidget(tilt_cb)
    left.addWidget(ctl)

    # 건물 보정
    bg = QtWidgets.QGroupBox("건물 보정 (지반 흔들림 역산)")
    bl = QtWidgets.QGridLayout(bg)
    bld_cb = QtWidgets.QCheckBox("건물 증폭을 빼고 지반 기준으로 판정"); bld_cb.setChecked(proc.bld["on"])
    top_sb = QtWidgets.QSpinBox(); top_sb.setRange(2, 200); top_sb.setValue(proc.bld["top"])
    cur_sb = QtWidgets.QSpinBox(); cur_sb.setRange(1, 200); cur_sb.setValue(proc.bld["cur"])
    t1_sb = QtWidgets.QDoubleSpinBox(); t1_sb.setRange(0.1, 10); t1_sb.setSingleStep(0.05); t1_sb.setDecimals(2)
    t1_sb.setValue(proc.bld_T1())
    z_sb = QtWidgets.QDoubleSpinBox(); z_sb.setRange(1, 20); z_sb.setSingleStep(0.5); z_sb.setValue(proc.bld["zeta"])
    t1_meas = QtWidgets.QPushButton("고유주기 측정")
    t1_est = QtWidgets.QPushButton("층수로 다시 추정")
    bl.addWidget(bld_cb, 0, 0, 1, 4)
    for r, (name, w) in enumerate([("최고 층수", top_sb), ("지금 층", cur_sb), ("고유주기 T₁ (초)", t1_sb), ("감쇠비 (%)", z_sb)]):
        bl.addWidget(QtWidgets.QLabel(name), 1 + r // 2, (r % 2) * 2); bl.addWidget(w, 1 + r // 2, (r % 2) * 2 + 1)
    bl.addWidget(t1_meas, 3, 0, 1, 2); bl.addWidget(t1_est, 3, 2, 1, 2)
    bld_info = label("", 11)
    bl.addWidget(bld_info, 4, 0, 1, 4)
    left.addWidget(bg)

    status = label("", 11)
    left.addWidget(status)
    log_list = QtWidgets.QListWidget(); log_list.setMaximumHeight(140)
    left.addWidget(QtWidgets.QLabel("감지 기록 (IMM III 이상, 장주기 계급 1 이상)"))
    left.addWidget(log_list)
    left.addStretch(1)

    # 그래프
    spec_plot = pg.PlotWidget(title="주기별 진도")
    spec_plot.setLogMode(x=True, y=False)
    spec_plot.setYRange(0.5, 10.5, padding=0)
    spec_plot.showGrid(x=True, y=True, alpha=0.3)
    spec_plot.setLabel("bottom", "목표 주기 T (초)")
    spec_plot.getAxis("left").setTicks([[(i + 1, c) for i, c in enumerate(CLASSES)]])
    for c in range(len(CLASSES)):   # 등급 띠
        band = pg.LinearRegionItem(values=(c + 0.5, c + 1.5), orientation="horizontal", movable=False,
                                   brush=pg.mkBrush(QtGui_color(MMI_COLORS[c], 60)), pen=pg.mkPen(None))
        spec_plot.addItem(band)
    noise_curve = spec_plot.plot(pen=pg.mkPen((120, 120, 120), width=1), fillLevel=0.5, brush=(120, 120, 120, 80))
    hold_curve = spec_plot.plot(pen=pg.mkPen((90, 105, 119), width=1.5, style=QtCore.Qt.PenStyle.DashLine))
    raw_curve = spec_plot.plot(pen=pg.mkPen((90, 105, 119), width=1))
    cur_curve = spec_plot.plot(pen=pg.mkPen((24, 35, 45), width=2.5))
    best_dot = spec_plot.plot(pen=None, symbol="o", symbolSize=10, symbolBrush=(24, 35, 45))
    t1_line = pg.InfiniteLine(angle=90, pen=pg.mkPen((31, 79, 184), style=QtCore.Qt.PenStyle.DotLine))
    spec_plot.addItem(t1_line)
    wave_plot = pg.PlotWidget(title="흔들림 (3성분 합성, gal)")
    wave_plot.showGrid(y=True, alpha=0.3)
    wave_curve = wave_plot.plot(pen=pg.mkPen((24, 35, 45), width=1))
    right.addWidget(spec_plot, 3)
    right.addWidget(wave_plot, 1)

    # ---------- 상태
    st = {"hold": None, "hold_sva": (0.0, 0.0), "active": None, "quiet": None, "calib": None,
          "rec": None, "log": [], "q": queue.Queue(), "client": None, "csv": None, "last_count": 0,
          "share": FirebaseShare(args.share) if args.share else None}
    if st["share"]:
        st["share"].start()

    def masked_noise():
        return None if (st["calib"] is not None or st["csv"]) else proc.cur_noise()

    def add_log(text):
        log_list.insertItem(0, text)

    def rebuild_and_reset():
        proc.rebuild(); st["hold"] = None; st["hold_sva"] = (0.0, 0.0); save_settings(proc); update_bld_info()

    def update_bld_info():
        T1 = proc.bld_T1()
        amp = abs(building_H(np.array([1 / T1]), proc.bld["top"], proc.bld["cur"], T1, proc.bld["zeta"])[0])
        txt = f"T₁ {T1:.2f}초 ({'입력·측정값' if proc.bld['T1'] else '층수 기준식'}). 이 층에서 T₁ 흔들림은 지반의 약 {amp:.1f}배."
        if proc.bld["on"] and proc.noise and proc.noise["bldKey"] != proc.bld_key():
            txt += " 건물 설정이 바뀌어 잡음 측정을 다시 해주세요."
        bld_info.setText(txt)

    def on_bld():
        proc.bld.update({"on": bld_cb.isChecked(), "top": top_sb.value(),
                         "cur": min(cur_sb.value(), top_sb.value()), "zeta": z_sb.value()})
        rebuild_and_reset()

    def on_t1():
        proc.bld["T1"] = t1_sb.value(); rebuild_and_reset()

    def on_t1_est():
        proc.bld["T1"] = None; t1_sb.blockSignals(True); t1_sb.setValue(proc.bld_T1()); t1_sb.blockSignals(False)
        rebuild_and_reset()

    def on_t1_measure():
        r = proc.measure_T1()
        if not r:
            bld_info.setText("뚜렷한 고유주기 봉우리를 찾지 못했어요. 20초 이상, 조용할 때 다시 해 보세요.")
            return
        proc.bld["T1"] = round(r["T"], 2)
        t1_sb.blockSignals(True); t1_sb.setValue(proc.bld["T1"]); t1_sb.blockSignals(False)
        rebuild_and_reset()
        bld_info.setText(f"측정한 고유주기 {proc.bld['T1']:.2f}초로 설정했어요 (선명도 {r['ratio']:.0f}배). " + bld_info.text())

    bld_cb.toggled.connect(on_bld); top_sb.valueChanged.connect(on_bld); cur_sb.valueChanged.connect(on_bld)
    z_sb.valueChanged.connect(on_bld); t1_sb.valueChanged.connect(on_t1)
    t1_est.clicked.connect(on_t1_est); t1_meas.clicked.connect(on_t1_measure)
    hc_cb.toggled.connect(lambda v: (setattr(proc, "hc", v), setattr(proc, "noise", None), rebuild_and_reset()))
    tilt_cb.toggled.connect(lambda v: (setattr(proc, "tilt", v), rebuild_and_reset()))
    reset_btn.clicked.connect(lambda: (st.update(hold=None, hold_sva=(0.0, 0.0), active=None)))

    def on_cal():
        st["calib"] = proc.total; cal_btn.setEnabled(False)
    cal_btn.clicked.connect(on_cal)

    def on_rec():
        if st["rec"]:
            st["rec"].close(); st["rec"] = None; rec_btn.setText("원시 기록 저장 시작"); return
        name = time.strftime("imm_raw_%Y%m%d_%H%M%S.csv")
        st["rec"] = open(name, "w", newline="", encoding="utf-8")
        st["rec"].write("time_s,ax_gal,ay_gal,az_gal,gx_rad_s,gy_rad_s,gz_rad_s\n")
        rec_btn.setText(f"저장 중지 ({name})")
    rec_btn.clicked.connect(on_rec)

    # ---------- 1초마다 판정
    def tick():
        if st["calib"] is not None and proc.total - st["calib"] >= CAL_SEC * FS:
            proc.calibrate(); save_settings(proc); st["calib"] = None; st["hold"] = None
            cal_btn.setEnabled(True); update_bld_info()
        res = proc.analyze()
        if not res:
            hero_sub.setText("데이터를 모으는 중이에요"); return
        prim = res["cor"] or res["raw"]
        nz = masked_noise()
        lv, best = proc.judge(prim["A"], nz)
        if st["share"] and st["calib"] is None:
            sv, sT = proc.share_values(prim, nz)
            st["share"].push(sv, sT)
        raw_lv = proc.judge(res["raw"]["A"], None if nz is None else proc.noise_set()["raw"])[0] if res["cor"] else None
        if st["hold"] is None:
            st["hold"] = list(lv)
        else:
            st["hold"] = [h if v is None else (v if h is None else max(h, v)) for h, v in zip(st["hold"], lv)]
        nsv = None if nz is None else proc.cur_noise("sva")
        lp_ok = not (nsv is not None and prim["sva"] < NOISE_MARGIN * nsv)
        lpc = lp_class(prim["sva"]) if lp_ok else 0
        if lp_ok and prim["sva"] > st["hold_sva"][0]:
            st["hold_sva"] = (prim["sva"], prim["svaT"])

        if best >= 0:
            c = class_of(lv[best])
            hero.setText(CLASSES[c])
            hero.setStyleSheet(f"background:{MMI_COLORS[c]}; color:{'#fff' if c >= 8 else '#14202a'};"
                               "border:1px solid #ccc; border-radius:12px; padding:8px;")
            sub = f"대표 주기 {TS[best]:.2f}초, 소수 진도 {fmt_frac(lv[best])}"
            if raw_lv is not None:
                rb = max((i for i, v in enumerate(raw_lv) if v is not None), key=lambda i: raw_lv[i], default=None)
                if rb is not None:
                    sub += f", {proc.bld['cur']}층 실측 {CLASSES[class_of(raw_lv[rb])]}"
            if proc.bld["on"]:
                sub = "지반 추정. " + sub
            if nz is None and not st["csv"]:
                sub += " (잡음 측정 전)"
            hero_sub.setText(sub)
        else:
            hero.setText("I"); hero.setStyleSheet("background:#fff; border:1px solid #ccc; border-radius:12px;")
            hero_sub.setText("모든 주기가 잡음 바닥 아래예요")
        refs.setText(f"PGA 기준 MMI {CLASSES[class_of(level_frac(prim['pga']))]} ({prim['pga']:.1f} gal)   "
                     f"JMA {prim['jma']:.1f} ({jma_label(prim['jma'])})   "
                     f"장주기 {'잡음' if not lp_ok else ('계급 ' + str(lpc) if lpc else '계급 1 미만')} "
                     f"(Sva {prim['sva']:.1f} cm/s, {prim['svaT']:.1f}초)")
        for i, s in enumerate(proc.sel):
            T, v, f = proc.eval_spec(s, prim["sel"][i], prim["A"], nz)
            at = "" if s[0] == s[1] else f" @ {T:.2f}초"
            sel_out[i].setText(("잡음" if f is None else f"{CLASSES[class_of(f)]}  {fmt_frac(f)}") + f"  ({v:.1f} gal{at})")

        x = np.log10(TS)
        ok = np.array([v is not None for v in lv])
        cur_curve.setData(TS[ok], np.array([v for v in lv if v is not None]))
        hold_ok = np.array([v is not None for v in st["hold"]])
        hold_curve.setData(TS[hold_ok], np.array([v for v in st["hold"] if v is not None]))
        raw_curve.setData(*(([], []) if raw_lv is None else
                            (TS[[v is not None for v in raw_lv]], np.array([v for v in raw_lv if v is not None]))))
        best_dot.setData([TS[best]] if best >= 0 else [], [min(lv[best], 10.5)] if best >= 0 else [])
        if nz is not None:
            noise_curve.setData(TS, [min(level_frac(NOISE_MARGIN * n), 10.5) for n in nz])
        else:
            noise_curve.setData([], [])
        t1_line.setVisible(proc.bld["on"]); t1_line.setValue(math.log10(proc.bld_T1()))
        pm = prim["pga_mag"]
        wave_plot.setTitle(f"흔들림 (3성분 합성, gal), 최대 {prim['pga']:.1f}{' (지반 추정)' if proc.bld['on'] else ''}")
        wave_curve.setData(np.arange(len(pm)) / FS - len(pm) / FS, pm)

        # 감지 기록
        cur = class_of(lv[best]) if best >= 0 else 0
        if st["calib"] is None and (cur >= LOG_FROM or lpc >= 1):
            st["quiet"] = None
            if st["active"] is None or cur > st["active"]["c"] or lpc > st["active"]["lp"]:
                where = f"CSV {st['csv']['cursor']:.0f}초" if st["csv"] else time.strftime("%Y-%m-%d %H:%M:%S")
                text = (f"{where}  IMM {CLASSES[cur]} (T* {TS[best]:.2f}초)  PGA {prim['pga']:.1f} gal  "
                        f"JMA {prim['jma']:.1f}  장주기 계급 {lpc}{'  지반 추정' if proc.bld['on'] else ''}")
                if st["active"] is None:
                    add_log(text)
                    with open("imm_events.csv", "a", encoding="utf-8") as f:
                        f.write(text.replace("  ", ",") + "\n")
                else:
                    log_list.item(0).setText(text)
                st["active"] = {"c": max(cur, (st["active"] or {"c": 0})["c"]),
                                "lp": max(lpc, (st["active"] or {"lp": 0})["lp"])}
        elif st["active"] is not None:
            if st["quiet"] is None:
                st["quiet"] = time.time()
            elif time.time() - st["quiet"] > 30:
                st["active"] = None; st["quiet"] = None

        hb = max((i for i, v in enumerate(st["hold"]) if v is not None), key=lambda i: st["hold"][i], default=None)
        nset = proc.noise_set()
        noise_txt = ("측정 중" if st["calib"] is not None else
                     ("없음 (잡음 측정 필요)" if nz is None and not st["csv"] else
                      ("CSV 재생 중엔 쓰지 않음" if st["csv"] else
                       f"1초 대역 {noise_at(nz, 1):.2f} gal, 10초 대역 {nz[-1]:.2f} gal")))
        if nset and proc.noise.get("on"):
            noise_txt += f"  |  5초 대역 자이로 보정 끔 {noise_at(proc.noise['off']['raw'], 5):.2f} → 켬 {noise_at(proc.noise['on']['raw'], 5):.2f} gal"
        tilt_txt = ("자이로 없음" if proc.gyro_count == 0 else
                    (f"보정량 최대 {res['tilt']['peak']:.2f} gal, 폰 기울기 {res['tilt']['deg']:.1f}°" if res["tilt"] else "보정 꺼짐"))
        src = (f"CSV {os.path.basename(args.csv)} 재생" if st["csv"] else
               (st["client"].status if st["client"] else ""))
        status.setText(f"입력: {src}, {st['rate']} Hz" + (f"  |  {st['share'].status}" if st["share"] else "")
                       + f"\n잡음 바닥: {noise_txt}\n기울어짐: {tilt_txt}\n"
                       f"기록 최대: " + (f"{CLASSES[class_of(st['hold'][hb])]} ({TS[hb]:.2f}초, {fmt_frac(st['hold'][hb])})" if hb is not None else "–")
                       + (f", 장주기 계급 {lp_class(st['hold_sva'][0])} (Sva {st['hold_sva'][0]:.1f})" if st["hold_sva"][0] else ""))

    st["rate"] = 0

    # ---------- 입력 공급
    if args.csv:
        t, acc, gyro = load_csv(args.csv, args.unit, args.fs, args.layout)
        st["csv"] = {"t": t, "acc": acc, "gyro": gyro, "i": 0, "cursor": 0.0}

        def feed_csv():
            c = st["csv"]
            if c is None:
                return
            upto = c["cursor"] + 1
            n0 = c["i"]
            while c["i"] < len(c["t"]) and c["t"][c["i"]] < upto:
                i = c["i"]
                g = c["gyro"][i] if c["gyro"] is not None else (None, None, None)
                proc.push(1000 + c["t"][i], *c["acc"][i], *g); c["i"] += 1
            if c["i"] >= len(c["t"]):
                for tt in np.arange(max(c["cursor"], c["t"][-1]) + 1 / FS, upto, 1 / FS):
                    proc.push(1000 + tt, *c["acc"][-1])
            st["rate"] = c["i"] - n0
            c["cursor"] = upto
            tick()
            if c["cursor"] >= c["t"][-1] + 5:
                timer.stop(); hero_sub.setText(hero_sub.text() + "  (재생 끝)")
        timer = QtCore.QTimer(); timer.timeout.connect(feed_csv)
        timer.start(0 if args.speed == "max" else int(1000 / float(args.speed)))
    else:
        st["client"] = make_client(args, st["q"])
        st["client"].start()

        def feed_live():
            n = 0
            while not st["q"].empty():
                s = st["q"].get()
                proc.push(*s); n += 1
                if st["rec"]:
                    g = s[4:] if s[4] is not None else ("", "", "")
                    st["rec"].write(",".join(str(v) for v in (s[0], s[1], s[2], s[3], *g)) + "\n")
            st["rate"] = n
            tick()
        timer = QtCore.QTimer(); timer.timeout.connect(feed_live); timer.start(1000)

    update_bld_info()
    win.resize(1280, 820)
    win.show()
    pg.exec()
    if st["client"]:
        st["client"].stop()
    if st["share"]:
        st["share"].stop()
    if st["rec"]:
        st["rec"].close()


def QtGui_color(hex_color, alpha):
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


# ===================================================================== 설정 저장
def save_settings(proc):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({"hc": proc.hc, "tilt": proc.tilt, "sel": proc.sel, "bld": proc.bld, "noise": proc.noise},
                      f, ensure_ascii=False)
    except OSError:
        pass


def load_settings(proc):
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            s = json.load(f)
    except (OSError, ValueError):
        return
    proc.hc = bool(s.get("hc", False))
    proc.tilt = bool(s.get("tilt", False))
    if isinstance(s.get("sel"), list) and len(s["sel"]) == 3:
        proc.sel = [tuple(x) for x in s["sel"]]
    if isinstance(s.get("bld"), dict):
        proc.bld.update(s["bld"])
    n = s.get("noise")
    if n and n.get("hc") == proc.hc and n.get("off") and len(n["off"]["raw"]) == NT:
        proc.noise = n
    proc.rebuild()


# ===================================================================== 시작
def main():
    ap = argparse.ArgumentParser(description="SensorServer 폰 센서로 IMM 진도를 실시간 계산")
    ap.add_argument("--host", default="192.168.0.10", help="SensorServer가 표시하는 폰 IP")
    ap.add_argument("--port", type=int, default=8080, help="SensorServer 포트")
    ap.add_argument("--no-gyro", action="store_true", help="자이로 없이 가속도만 받기")
    ap.add_argument("--serial", help="아두이노 시리얼 포트 (예: COM3, /dev/ttyACM0). 주면 SensorServer 대신 사용")
    ap.add_argument("--baud", type=int, default=230400, help="시리얼 속도 (스케치와 같아야 함)")
    ap.add_argument("--scale", type=float, default=None, help="gal/카운트 직접 지정 (생략하면 중력으로 자동 보정)")
    ap.add_argument("--csv", help="실시간 대신 CSV 파형 재생")
    ap.add_argument("--unit", default="gal", choices=["gal", "ms2", "g"], help="CSV 가속도 단위")
    ap.add_argument("--fs", type=float, default=100.0, help="CSV에 시간 열이 없을 때 샘플링 (Hz)")
    ap.add_argument("--layout", default="auto", choices=["auto", "txyz", "xyz"], help="CSV 열 구성")
    ap.add_argument("--speed", default="5", help="CSV 재생 속도: 1, 2, 5, max")
    ap.add_argument("--nogui", action="store_true", help="창 없이 콘솔에 1초마다 출력")
    ap.add_argument("--share", help="IMM 관측 지도 공유 설정 파일 (JSON). 주면 1초마다 관측값을 올림")
    args = ap.parse_args()

    if args.csv and args.share:
        print("CSV 재생 값은 관측 지도에 올리지 않아요. --share를 무시합니다.")
        args.share = None
    proc = IMMProcessor()
    load_settings(proc)
    if args.nogui:
        run_headless(args, proc)
    else:
        run_gui(args, proc)


if __name__ == "__main__":
    main()
