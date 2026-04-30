
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.io.wavfile import read
from scipy.signal import resample

import numpy as np
import pandas as pd

Array = np.ndarray


@dataclass
class HarnessConfig:
    fs: int = 16000
    block_len: int = 128
    diag_fft_len: int = 256
    erle_avg_blocks: int = 64
    bypass_dtd: bool = False
    include_road_noise_in_rollup: bool = False
    duration_s: float = 8.0

    dt_enter: float = 0.50
    dt_exit: float = 0.30
    hold_blocks: int = 10
    ref_power_floor: float = 1e-8

    # Backend-aware adaptation controller
    backend_control_relaxation: float = 1.0
    backend_hold_scale: float = 1.0

    convergence_target_db: float = 20.0
    convergence_hold_blocks: int = 8
    convergence_start_mode: str = "far_end_onset"
    convergence_manual_start_block: int = 0

    pbfdaf_n_partitions: int = 5
    pbfdaf_mu: float = 0.10
    pbfdaf_power_smooth: float = 0.20
    pbfdaf_delta: float = 1e-9

    mdf_n_partitions: int = 5
    mdf_mu: float = 0.12
    mdf_power_smooth: float = 0.20
    mdf_delta: float = 1e-9
    mdf_leakage: float = 1.0

    nlms_filter_len: int = 640
    nlms_mu: float = 0.85
    nlms_delta: float = 1e-8
    nlms_leakage: float = 1.0
    nlms_prewhiten_ref: bool = True

    sbnlms_filter_len: int = 640
    sbnlms_num_taps: int = 5
    sbnlms_mu: float = 0.12
    sbnlms_power_smooth: float = 0.20
    sbnlms_delta: float = 1e-9

    fblms_filter_len: int = 640
    fblms_mu: float = 0.12
    fblms_power_smooth: float = 0.20
    fblms_delta: float = 1e-9

    apa_filter_len: int = 640
    apa_mu: float = 0.5
    apa_delta: float = 1e-3
    apa_order: int = 4

    # Fast Filtered-X APA
    # Default secondary path is identity for this AEC harness.
    fxapa_filter_len: int = 640
    fxapa_mu: float = 0.45
    fxapa_delta: float = 1e-3
    fxapa_order: int = 4
    fxapa_secondary_delay: int = 0
    fxapa_secondary_gain: float = 1.0

    # Fast APA approximation using Toeplitz autocorrelation + Levinson solve.
    # Avoids explicit projection matrix inversion / generic KxK solve per sample.
    fapa_filter_len: int = 640
    fapa_mu: float = 0.50
    fapa_delta: float = 1e-3
    fapa_order: int = 4
    fapa_cg_iters: int = 8
    fapa_update_clip: float = 1e9


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def colored_noise(n: int, color: str = "white", seed: int = 0) -> Array:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    if color == "white":
        return x.astype(np.float64)
    if color == "brown":
        y = np.cumsum(x)
        y = y / (np.std(y) + 1e-12)
        return y.astype(np.float64)
    if color == "pinkish":
        y = x.copy()
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
        for _ in range(3):
            y = np.convolve(y, kernel, mode="same")
        y = y / (np.std(y) + 1e-12)
        return y.astype(np.float64)
    raise ValueError(color)


def make_cabin_ir() -> Array:
    ir = np.zeros(640, dtype=np.float64)
    taps = [20, 40, 75, 120, 200, 310, 430, 560]
    amps = [0.85, 0.30, 0.18, 0.10, 0.06, 0.04, 0.03, 0.02]
    for t, a in zip(taps, amps):
        ir[t] = a
    return ir


def apply_nonlinear_playback(x: Array, drive: float = 3.0) -> Array:
    return (np.tanh(drive * x) / np.tanh(drive)).astype(np.float64)


def sample_to_block_labels(labels: Array, block_len: int) -> Array:
    n_blocks = len(labels) // block_len
    y = np.zeros(n_blocks, dtype=np.float64)
    for b in range(n_blocks):
        s = b * block_len
        e = s + block_len
        y[b] = 1.0 if np.mean(labels[s:e]) > 0.25 else 0.0
    return y


def simulate_scenario(name: str, fs: int = 16000, dur_s: float = 8.0, seed: int = 0) -> Dict[str, Array]:
    n = int(fs * dur_s)
    ref = 0.1 * colored_noise(n, color="pinkish", seed=seed + 10)
    near = np.zeros(n, dtype=np.float64)
    additive_noise = np.zeros(n, dtype=np.float64)
    dt_truth = np.zeros(n, dtype=np.float64)
    ir = make_cabin_ir()

    if name == "far_end_only":
        pass
    elif name == "balanced_double_talk":
        s0, s1 = int(2.0 * fs), int(5.0 * fs)
        near[s0:s1] = 0.05 * colored_noise(s1 - s0, color="white", seed=seed + 1)
        dt_truth[s0:s1] = 1.0
    elif name == "near_end_dominant":
        s0, s1 = int(2.0 * fs), int(5.0 * fs)
        near[s0:s1] = 0.09 * colored_noise(s1 - s0, color="white", seed=seed + 2)
        dt_truth[s0:s1] = 1.0
    elif name == "path_variability":
        s0, s1 = int(2.0 * fs), int(5.0 * fs)
        near[s0:s1] = 0.05 * colored_noise(s1 - s0, color="white", seed=seed + 3)
        dt_truth[s0:s1] = 1.0
        change_idx = int(4.0 * fs)
        ir2 = np.roll(ir, 8)
        echo0 = np.convolve(ref[:change_idx], ir, mode="full")[:change_idx]
        echo1 = np.convolve(ref[change_idx:], ir2, mode="full")[:n - change_idx]
        mic = np.concatenate([echo0, echo1]) + near
        return {"ref": ref, "mic": mic.astype(np.float64), "dt_truth": dt_truth, "ir": ir}
    elif name == "nonlinear_playback":
        s0, s1 = int(2.0 * fs), int(5.0 * fs)
        near[s0:s1] = 0.05 * colored_noise(s1 - s0, color="white", seed=seed + 4)
        dt_truth[s0:s1] = 1.0
        ref = apply_nonlinear_playback(ref, drive=3.0)
    elif name == "road_noise_robustness":
        s0, s1 = int(2.0 * fs), int(5.0 * fs)
        near[s0:s1] = 0.05 * colored_noise(s1 - s0, color="white", seed=seed + 5)
        dt_truth[s0:s1] = 1.0
        additive_noise = 0.03 * colored_noise(n, color="brown", seed=seed + 6)
    elif name == "delay_jump":
        s0, s1 = int(2.0 * fs), int(5.0 * fs)
        near[s0:s1] = 0.05 * colored_noise(s1 - s0, color="white", seed=seed + 7)
        dt_truth[s0:s1] = 1.0
        j = int(4.0 * fs)
        ref2 = np.concatenate([np.zeros(16, dtype=np.float64), ref[:-16]])
        echo0 = np.convolve(ref[:j], ir, mode="full")[:j]
        echo1 = np.convolve(ref2[j:], ir, mode="full")[:n - j]
        mic = np.concatenate([echo0, echo1]) + near
        return {"ref": ref, "mic": mic.astype(np.float64), "dt_truth": dt_truth, "ir": ir}
    elif name == "voices":
        Fs, farend_data = read('data\\Hill_noisy.wav')
        ref = farend_data/32768.0
        if Fs != fs:
            ref = resample(ref,int(len(ref)*fs/Fs))
        Fs2,nearend_data = read('data\\armstrong_noisy.wav')
        near = nearend_data / 32768.0
        if Fs2 != fs:
            near = resample(near,int(len(near)*fs/Fs2))
        L = max(len(ref),len(near))
        ref = ref[:L]
        near = near[:L]
        additive_noise = np.zeros(L, dtype=np.float64)
        dt_truth = np.zeros(L, dtype=np.float64)
        #additive_noise = 0.03 * colored_noise(L, color="brown", seed=seed + 6)
        echo = np.convolve(ref, ir, mode="full")[:L]
        mic = echo + near + additive_noise
        return {"ref": ref, "mic": mic.astype(np.float64), "dt_truth": dt_truth, "ir": ir}
    else:
        raise ValueError(name)

    echo = np.convolve(ref, ir, mode="full")[:n]
    mic = echo + near + additive_noise
    return {"ref": ref.astype(np.float64), "mic": mic.astype(np.float64), "dt_truth": dt_truth, "ir": ir}


def erle_db(mic: Array, err: Array, start_idx: int = 0) -> float:
    return float(10.0 * np.log10((np.mean(mic[start_idx:] ** 2) + 1e-18) / (np.mean(err[start_idx:] ** 2) + 1e-18)))


def onset_delay_ms(gt: Array, pred: Array, block_len: int, fs: int) -> float:
    gt_idx = np.flatnonzero((gt[1:] > gt[:-1]) & (gt[1:] > 0)) + 1
    if len(gt_idx) == 0:
        return np.nan
    vals = []
    for idx in gt_idx:
        hit = np.flatnonzero(pred[idx:] > 0)
        if len(hit):
            vals.append(hit[0] * block_len * 1000.0 / fs)
    return float(np.mean(vals)) if vals else np.nan


def release_delay_ms(gt: Array, pred: Array, block_len: int, fs: int) -> float:
    gt_idx = np.flatnonzero((gt[1:] < gt[:-1]) & (gt[:-1] > 0)) + 1
    if len(gt_idx) == 0:
        return np.nan
    vals = []
    for idx in gt_idx:
        hit = np.flatnonzero(pred[idx:] <= 0)
        if len(hit):
            vals.append(hit[0] * block_len * 1000.0 / fs)
    return float(np.mean(vals)) if vals else np.nan


def compute_erle_series(mic: Array, err: Array, cfg: HarnessConfig) -> Tuple[Array, Array]:
    n_blocks = min(len(mic), len(err)) // cfg.block_len
    raw_erle = np.zeros(n_blocks, dtype=np.float64)
    avg_erle = np.zeros(n_blocks, dtype=np.float64)
    avg_blocks = max(1, int(cfg.erle_avg_blocks))
    for i in range(n_blocks):
        s = i * cfg.block_len
        e = s + cfg.block_len
        raw_erle[i] = erle_db(mic[s:e], err[s:e], 0)
        s0 = max(0, (i - avg_blocks + 1) * cfg.block_len)
        avg_erle[i] = erle_db(mic[s0:e], err[s0:e], 0)
    return raw_erle, avg_erle


def find_convergence_start_block(scenario_name: str, cfg: HarnessConfig) -> int:
    if cfg.convergence_start_mode == "manual_block":
        return int(cfg.convergence_manual_start_block)
    if cfg.convergence_start_mode == "path_change" and scenario_name in ("path_variability", "delay_jump"):
        return int((4.0 * cfg.fs) // cfg.block_len)
    return 0


def convergence_time_ms(avg_erle: Array, scenario_name: str, cfg: HarnessConfig) -> Tuple[float, bool]:
    start_block = find_convergence_start_block(scenario_name, cfg)
    hold = max(1, int(cfg.convergence_hold_blocks))
    target = float(cfg.convergence_target_db)
    if start_block >= len(avg_erle):
        return np.nan, False
    hits = avg_erle >= target
    for i in range(start_block, len(avg_erle) - hold + 1):
        if np.all(hits[i:i + hold]):
            return float((i - start_block) * cfg.block_len * 1000.0 / cfg.fs), True
    return np.nan, False


def compute_metrics(scenario_name: str, mic: Array, err: Array, p_dt_smooth: Array, dt_truth: Array, cfg: HarnessConfig, threshold: float = 0.5) -> Dict[str, float]:
    gt = sample_to_block_labels(dt_truth, cfg.block_len)
    pred = (p_dt_smooth >= threshold).astype(np.float64)
    dt_mask = gt > 0
    not_dt_mask = ~dt_mask
    late = int(0.75 * len(mic))
    _, avg_erle = compute_erle_series(mic, err, cfg)
    conv_ms, converged = convergence_time_ms(avg_erle, scenario_name, cfg)
    return {
        "threshold": float(threshold),
        "dt_miss_rate": float(np.mean(pred[dt_mask] == 0)) if np.any(dt_mask) else np.nan,
        "dt_false_alarm_rate": float(np.mean(pred[not_dt_mask] == 1)) if np.any(not_dt_mask) else np.nan,
        "onset_delay_ms": onset_delay_ms(gt, pred, cfg.block_len, cfg.fs),
        "release_delay_ms": release_delay_ms(gt, pred, cfg.block_len, cfg.fs),
        "mean_dt_prob": float(np.mean(p_dt_smooth)),
        "erle_db_full": erle_db(mic, err, 0),
        "erle_db_late": erle_db(mic, err, late),
        "convergence_time_ms": conv_ms,
        "converged_to_target": bool(converged),
    }




def plot_scenario(out_dir: Path, name: str, ref: Array, mic: Array, err: Array, p_dt: Array, mu_scale: Array, state_trace: Array, cfg: HarnessConfig, dt_truth: Array) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(mic)) / cfg.fs
    tb = np.arange(len(p_dt)) * cfg.block_len / cfg.fs
    gt = sample_to_block_labels(dt_truth, cfg.block_len)
    raw_erle, avg_erle = compute_erle_series(mic, err, cfg)
    conv_ms, converged = convergence_time_ms(avg_erle, name, cfg)

    fig = plt.figure(figsize=(10, 9))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(t, ref, label="ref")
    ax1.plot(t, mic, label="mic", alpha=0.8)
    ax1.plot(t, err, label="err", alpha=0.8)
    ax1.legend(loc="upper right")
    ax1.set_title(f"{name} signals")
    ax1.grid(True)

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(tb, p_dt, label="p_dt_smooth")
    ax2.plot(tb, gt, label="dt_truth", alpha=0.8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("DTD probability + state")
    ax2.grid(True)

    ax2b = ax2.twinx()
    ax2b.step(tb, state_trace, where="post", label="state", alpha=0.8)
    ax2b.set_ylim(-0.5, 4.5)
    ax2b.set_yticks([0, 1, 2, 3, 4])
    ax2b.set_yticklabels(["idle", "double_talk", "near_end_only", "far_end_only", "hold"])

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(tb, mu_scale)
    ax3.set_title("Adaptation scale")
    ax3.grid(True)

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(tb, raw_erle, label="raw block ERLE")
    ax4.plot(tb, avg_erle, label=f"moving-average ERLE ({int(cfg.erle_avg_blocks)} blocks)")
    title = "ERLE"
    if converged:
        title += f" | conv={conv_ms:.1f} ms"
    ax4.set_title(title)
    ax4.legend(loc="upper right")
    ax4.grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def block_diagnostic_ffts(prev_ref: Array, ref_block: Array, mic_block: Array, err_block: Array, echo_block: Array, fft_len: int) -> Dict[str, Array]:
    z = np.zeros_like(ref_block)
    ref_fft = np.fft.rfft(np.concatenate([prev_ref, ref_block]).astype(np.float64), n=fft_len)
    mic_fft = np.fft.rfft(np.concatenate([z, mic_block]).astype(np.float64), n=fft_len)
    err_fft = np.fft.rfft(np.concatenate([z, err_block]).astype(np.float64), n=fft_len)
    echo_fft = np.fft.rfft(np.concatenate([z, echo_block]).astype(np.float64), n=fft_len)
    return {"ref_fft": ref_fft, "mic_fft": mic_fft, "err_fft": err_fft, "echo_fft": echo_fft}


class AECBase:
    name = "base"
    def process_block(self, ref_block: Array, mic_block: Array, mu_scale: float = 1.0) -> Dict[str, Array]:
        raise NotImplementedError


class VerifiedPBFDAF(AECBase):
    name = "pbfdaf"
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.L = cfg.block_len
        self.N = 2 * cfg.block_len
        self.K = self.N // 2 + 1
        self.P = cfg.pbfdaf_n_partitions
        self.H = np.zeros((self.P, self.K), dtype=np.complex128)
        self.X_hist = np.zeros((self.P, self.K), dtype=np.complex128)
        self.prev_ref = np.zeros(self.L, dtype=np.float64)
        self.pow_s = np.ones(self.K, dtype=np.float64) * cfg.pbfdaf_delta

    def process_block(self, ref_block: Array, mic_block: Array, mu_scale: float = 1.0) -> Dict[str, Array]:
        ref_fft = np.fft.rfft(np.concatenate([self.prev_ref, ref_block]), n=self.N)
        self.X_hist[1:] = self.X_hist[:-1]
        self.X_hist[0] = ref_fft
        echo_fft_full = np.sum(self.H * self.X_hist, axis=0)
        echo_time_full = np.fft.irfft(echo_fft_full, n=self.N)
        echo_block = echo_time_full[self.L:]
        err_block = mic_block - echo_block
        power = np.sum((self.X_hist * np.conj(self.X_hist)).real, axis=0)
        self.pow_s = self.cfg.pbfdaf_power_smooth * self.pow_s + (1.0 - self.cfg.pbfdaf_power_smooth) * power
        err_fft = np.fft.rfft(np.concatenate([np.zeros(self.L, dtype=np.float64), err_block]), n=self.N)
        G = err_fft / (self.pow_s + self.cfg.pbfdaf_delta)
        mu_eff = self.cfg.pbfdaf_mu * float(mu_scale)
        for p in range(self.P):
            grad = np.conj(self.X_hist[p]) * G
            h_time = np.fft.irfft(self.H[p] + mu_eff * grad, n=self.N)
            h_time[self.L:] = 0.0
            self.H[p] = np.fft.rfft(h_time, n=self.N)
        diag = block_diagnostic_ffts(self.prev_ref, ref_block, mic_block, err_block, echo_block, self.cfg.diag_fft_len)
        self.prev_ref = ref_block.copy()
        return {"echo_block": echo_block, "err_block": err_block, **diag}


class MDFEchoCanceller(AECBase):
    name = "mdf"
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.L = cfg.block_len
        self.N = 2 * cfg.block_len
        self.K = self.N // 2 + 1
        self.P = cfg.mdf_n_partitions
        self.H = np.zeros((self.P, self.K), dtype=np.complex128)
        self.X_hist = np.zeros((self.P, self.K), dtype=np.complex128)
        self.prev_ref = np.zeros(self.L, dtype=np.float64)
        self.pow_s = np.ones(self.K, dtype=np.float64) * cfg.mdf_delta

    def process_block(self, ref_block: Array, mic_block: Array, mu_scale: float = 1.0) -> Dict[str, Array]:
        ref_fft = np.fft.rfft(np.concatenate([self.prev_ref, ref_block]), n=self.N)
        self.X_hist[1:] = self.X_hist[:-1]
        self.X_hist[0] = ref_fft
        echo_fft_full = np.sum(self.H * self.X_hist, axis=0)
        echo_time_full = np.fft.irfft(echo_fft_full, n=self.N)
        echo_block = echo_time_full[self.L:]
        err_block = mic_block - echo_block
        err_fft = np.fft.rfft(np.concatenate([np.zeros(self.L, dtype=np.float64), err_block]), n=self.N)
        power = np.sum((self.X_hist * np.conj(self.X_hist)).real, axis=0)
        self.pow_s = self.cfg.mdf_power_smooth * self.pow_s + (1.0 - self.cfg.mdf_power_smooth) * power
        mu_eff = self.cfg.mdf_mu * float(mu_scale)
        for p in range(self.P):
            grad = np.conj(self.X_hist[p]) * err_fft / (self.pow_s + self.cfg.mdf_delta)
            h_time = np.fft.irfft(self.H[p], n=self.N)
            h_time[:self.L] *= self.cfg.mdf_leakage
            h_time[self.L:] = 0.0
            grad_time = np.fft.irfft(grad, n=self.N)
            grad_time[self.L:] = 0.0
            h_time = h_time + mu_eff * grad_time
            h_time[self.L:] = 0.0
            self.H[p] = np.fft.rfft(h_time, n=self.N)
        diag = block_diagnostic_ffts(self.prev_ref, ref_block, mic_block, err_block, echo_block, self.cfg.diag_fft_len)
        self.prev_ref = ref_block.copy()
        return {"echo_block": echo_block, "err_block": err_block, **diag}


class NLMSEchoCanceller(AECBase):
    name = "nlms"
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.M = cfg.nlms_filter_len
        self.w = np.zeros(self.M, dtype=np.float64)
        self.xbuf = np.zeros(self.M, dtype=np.float64)
        self.prev_ref = np.zeros(cfg.block_len, dtype=np.float64)
        self.prev_sample = 0.0

    def process_block(self, ref_block: Array, mic_block: Array, mu_scale: float = 1.0) -> Dict[str, Array]:
        L = len(ref_block)
        echo_block = np.zeros(L, dtype=np.float64)
        err_block = np.zeros(L, dtype=np.float64)
        mu_eff = self.cfg.nlms_mu * float(mu_scale)
        for n in range(L):
            x_in = ref_block[n]
            if self.cfg.nlms_prewhiten_ref:
                x_use = x_in - 0.85 * self.prev_sample
                self.prev_sample = x_in
            else:
                x_use = x_in
            self.xbuf[1:] = self.xbuf[:-1]
            self.xbuf[0] = x_use
            y = float(np.dot(self.w, self.xbuf))
            e = float(mic_block[n] - y)
            norm = self.cfg.nlms_delta + float(np.dot(self.xbuf, self.xbuf))
            self.w *= self.cfg.nlms_leakage
            self.w += (mu_eff / norm) * e * self.xbuf
            echo_block[n] = y
            err_block[n] = e
        diag = block_diagnostic_ffts(self.prev_ref, ref_block, mic_block, err_block, echo_block, self.cfg.diag_fft_len)
        self.prev_ref = ref_block.copy()
        return {"echo_block": echo_block, "err_block": err_block, **diag}




class SubbandNLMSEchoCanceller(AECBase):
    name = "sbnlms"

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.R = cfg.block_len                 # explicit decimation: one subband update per R fullband samples
        self.N = 2 * cfg.block_len            # WOLA frame length
        self.K = self.N // 2 + 1
        self.P = max(1, int(cfg.sbnlms_num_taps))

        self.win = np.sqrt(np.hanning(self.N + 1)[:-1]).astype(np.float64)

        self.ref_buf = np.zeros(self.N, dtype=np.float64)
        self.mic_buf = np.zeros(self.N, dtype=np.float64)

        self.H = np.zeros((self.K, self.P), dtype=np.complex128)
        self.X_hist = np.zeros((self.K, self.P), dtype=np.complex128)
        self.pow_s = np.ones(self.K, dtype=np.float64) * cfg.sbnlms_delta

        self.ola = np.zeros(self.N, dtype=np.float64)
        self.prev_ref = np.zeros(cfg.block_len, dtype=np.float64)

    def process_block(self, ref_block: Array, mic_block: Array, mu_scale: float = 1.0) -> Dict[str, Array]:
        # Analysis at the subband frame rate (decimated by hop R)
        self.ref_buf[:-self.R] = self.ref_buf[self.R:]
        self.ref_buf[-self.R:] = ref_block
        self.mic_buf[:-self.R] = self.mic_buf[self.R:]
        self.mic_buf[-self.R:] = mic_block

        X = np.fft.rfft(self.ref_buf * self.win, n=self.N)
        D = np.fft.rfft(self.mic_buf * self.win, n=self.N)

        self.X_hist[:, 1:] = self.X_hist[:, :-1]
        self.X_hist[:, 0] = X

        # Per-band short adaptive filters in subband time
        Y = np.sum(self.H * self.X_hist, axis=1)

        # Synthesis overlap-add back to fullband
        y_frame = np.fft.irfft(Y, n=self.N).real * self.win
        self.ola += y_frame
        echo_block = self.ola[:self.R].copy()
        self.ola[:-self.R] = self.ola[self.R:]
        self.ola[-self.R:] = 0.0

        err_block = mic_block - echo_block

        E = D - Y
        power = np.sum((self.X_hist * np.conj(self.X_hist)).real, axis=1)
        self.pow_s = self.cfg.sbnlms_power_smooth * self.pow_s + (1.0 - self.cfg.sbnlms_power_smooth) * power

        mu_eff = self.cfg.sbnlms_mu * float(mu_scale)
        for p in range(self.P):
            self.H[:, p] += mu_eff * (np.conj(self.X_hist[:, p]) * E / (self.pow_s + self.cfg.sbnlms_delta))

        diag = block_diagnostic_ffts(self.prev_ref, ref_block, mic_block, err_block, echo_block, self.cfg.diag_fft_len)
        self.prev_ref = ref_block.copy()
        return {"echo_block": echo_block, "err_block": err_block, **diag}


class FBLMSEchoCanceller(AECBase):
    name = "fblms"
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.L = cfg.block_len
        self.M = cfg.fblms_filter_len
        self.N = next_pow2(self.M + self.L - 1)
        self.K = self.N // 2 + 1
        self.W = np.zeros(self.K, dtype=np.complex128)
        self.x_hist = np.zeros(self.M - 1, dtype=np.float64)
        self.prev_ref = np.zeros(cfg.block_len, dtype=np.float64)
        self.pow_s = np.ones(self.K, dtype=np.float64) * cfg.fblms_delta

    def process_block(self, ref_block: Array, mic_block: Array, mu_scale: float = 1.0) -> Dict[str, Array]:
        frame = np.concatenate([self.x_hist, ref_block]).astype(np.float64)
        X = np.fft.rfft(frame, n=self.N)
        y_full = np.fft.irfft(self.W * X, n=self.N)
        echo_block = y_full[self.M - 1:self.M - 1 + self.L]
        err_block = mic_block - echo_block
        e_pad = np.concatenate([np.zeros(self.M - 1, dtype=np.float64), err_block])
        if len(e_pad) < self.N:
            e_pad = np.pad(e_pad, (0, self.N - len(e_pad)))
        E = np.fft.rfft(e_pad, n=self.N)
        power = (X * np.conj(X)).real
        self.pow_s = self.cfg.fblms_power_smooth * self.pow_s + (1.0 - self.cfg.fblms_power_smooth) * power
        G = np.conj(X) * E / (self.pow_s + self.cfg.fblms_delta)
        w_time = np.fft.irfft(self.W + self.cfg.fblms_mu * float(mu_scale) * G, n=self.N)
        w_time[self.M:] = 0.0
        self.W = np.fft.rfft(w_time, n=self.N)
        self.x_hist = frame[-(self.M - 1):].copy()
        diag = block_diagnostic_ffts(self.prev_ref, ref_block, mic_block, err_block, echo_block, self.cfg.diag_fft_len)
        self.prev_ref = ref_block.copy()
        return {"echo_block": echo_block, "err_block": err_block, **diag}


class APAEchoCanceller(AECBase):
    name = "apa"
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.M = cfg.apa_filter_len
        self.K = cfg.apa_order
        self.w = np.zeros(self.M, dtype=np.float64)
        self.xbuf = np.zeros(self.M, dtype=np.float64)
        self.prev_ref = np.zeros(cfg.block_len, dtype=np.float64)
        self.u_hist: List[np.ndarray] = []
        self.d_hist: List[float] = []

    def process_block(self, ref_block: Array, mic_block: Array, mu_scale: float = 1.0) -> Dict[str, Array]:
        L = len(ref_block)
        echo_block = np.zeros(L, dtype=np.float64)
        err_block = np.zeros(L, dtype=np.float64)
        mu_eff = self.cfg.apa_mu * float(mu_scale)
        for n in range(L):
            self.xbuf[1:] = self.xbuf[:-1]
            self.xbuf[0] = ref_block[n]
            u = self.xbuf.copy()
            d = float(mic_block[n])
            y = float(self.w @ u)
            e = d - y
            self.u_hist.insert(0, u)
            self.d_hist.insert(0, d)
            if len(self.u_hist) > self.K:
                self.u_hist.pop()
                self.d_hist.pop()
            U = np.stack(self.u_hist, axis=1)
            dvec = np.array(self.d_hist, dtype=np.float64)
            yvec = self.w @ U
            evec = dvec - yvec
            G = U.T @ U + self.cfg.apa_delta * np.eye(U.shape[1], dtype=np.float64)
            try:
                z = np.linalg.solve(G, evec)
            except np.linalg.LinAlgError:
                z = np.linalg.pinv(G) @ evec
            self.w = self.w + mu_eff * (U @ z)
            echo_block[n] = y
            err_block[n] = e
        diag = block_diagnostic_ffts(self.prev_ref, ref_block, mic_block, err_block, echo_block, self.cfg.diag_fft_len)
        self.prev_ref = ref_block.copy()
        return {"echo_block": echo_block, "err_block": err_block, **diag}



class FilteredXAPADirectEchoCanceller(AECBase):
    name = "fxapa_direct"

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.M = cfg.fxapa_filter_len
        self.K = cfg.fxapa_order
        self.w = np.zeros(self.M, dtype=np.float64)

        self.xbuf_raw = np.zeros(self.M, dtype=np.float64)
        self.xbuf_filt = np.zeros(self.M, dtype=np.float64)

        self.s_delay = max(0, int(cfg.fxapa_secondary_delay))
        self.s_gain = float(cfg.fxapa_secondary_gain)
        self.secondary_fifo = np.zeros(self.s_delay + 1, dtype=np.float64)

        self.prev_ref = np.zeros(cfg.block_len, dtype=np.float64)
        self.U_hist: List[np.ndarray] = []
        self.d_hist: List[float] = []
        self.eye_cache: Dict[int, np.ndarray] = {}

    def _filtered_x_sample(self, x: float) -> float:
        self.secondary_fifo[1:] = self.secondary_fifo[:-1]
        self.secondary_fifo[0] = x
        return self.s_gain * self.secondary_fifo[-1]

    def process_block(self, ref_block: Array, mic_block: Array, mu_scale: float = 1.0) -> Dict[str, Array]:
        L = len(ref_block)
        echo_block = np.zeros(L, dtype=np.float64)
        err_block = np.zeros(L, dtype=np.float64)
        mu_eff = self.cfg.fxapa_mu * float(mu_scale)

        for n in range(L):
            x_raw = float(ref_block[n])
            x_filt = self._filtered_x_sample(x_raw)

            self.xbuf_raw[1:] = self.xbuf_raw[:-1]
            self.xbuf_raw[0] = x_raw

            self.xbuf_filt[1:] = self.xbuf_filt[:-1]
            self.xbuf_filt[0] = x_filt

            y = float(self.w @ self.xbuf_raw)
            d = float(mic_block[n])
            e = d - y

            u = self.xbuf_filt.copy()
            self.U_hist.insert(0, u)
            self.d_hist.insert(0, d)
            if len(self.U_hist) > self.K:
                self.U_hist.pop()
                self.d_hist.pop()

            U = np.stack(self.U_hist, axis=1)
            dvec = np.array(self.d_hist, dtype=np.float64)
            yvec = self.w @ U
            evec = dvec - yvec

            kcur = U.shape[1]
            eye = self.eye_cache.get(kcur)
            if eye is None:
                eye = np.eye(kcur, dtype=np.float64)
                self.eye_cache[kcur] = eye

            G = U.T @ U + self.cfg.fxapa_delta * eye
            try:
                z = np.linalg.solve(G, evec)
            except np.linalg.LinAlgError:
                z = np.linalg.pinv(G) @ evec

            self.w = self.w + mu_eff * (U @ z)

            echo_block[n] = y
            err_block[n] = e

        diag = block_diagnostic_ffts(self.prev_ref, ref_block, mic_block, err_block, echo_block, self.cfg.diag_fft_len)
        self.prev_ref = ref_block.copy()
        return {"echo_block": echo_block, "err_block": err_block, **diag}






def solve_projection_cg(G: Array, b: Array, n_iter: int = 8, tol: float = 1e-10) -> Array:
    """Iterative projection solver for G z = b without inversion or np.linalg.solve."""
    n = len(b)
    z = np.zeros(n, dtype=np.float64)
    if n == 0:
        return z
    r = b.astype(np.float64).copy()
    p = r.copy()
    rsold = float(r @ r)
    if rsold < tol:
        return z
    for _ in range(max(1, int(n_iter))):
        Ap = G @ p
        denom = float(p @ Ap) + 1e-18
        alpha = rsold / denom
        z += alpha * p
        r -= alpha * Ap
        rsnew = float(r @ r)
        if rsnew < tol:
            break
        p = r + (rsnew / (rsold + 1e-18)) * p
        rsold = rsnew
    return z





class FastAPAEchoCanceller(AECBase):
    name = "fapa"

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.M = cfg.fapa_filter_len
        self.K = cfg.fapa_order
        self.w = np.zeros(self.M, dtype=np.float64)
        self.xbuf = np.zeros(self.M, dtype=np.float64)
        self.prev_ref = np.zeros(cfg.block_len, dtype=np.float64)
        self.U = np.zeros((self.M, self.K), dtype=np.float64)
        self.dvec = np.zeros(self.K, dtype=np.float64)
        self.kcur = 0

    def process_block(self, ref_block: Array, mic_block: Array, mu_scale: float = 1.0) -> Dict[str, Array]:
        L = len(ref_block)
        echo_block = np.zeros(L, dtype=np.float64)
        err_block = np.zeros(L, dtype=np.float64)
        mu_eff = self.cfg.fapa_mu * float(mu_scale)

        for n in range(L):
            self.xbuf[1:] = self.xbuf[:-1]
            self.xbuf[0] = float(ref_block[n])

            u = self.xbuf
            d = float(mic_block[n])
            y = float(self.w @ u)
            e = d - y

            # Projection history; column 0 is newest regressor.
            self.U[:, 1:] = self.U[:, :-1]
            self.U[:, 0] = u
            self.dvec[1:] = self.dvec[:-1]
            self.dvec[0] = d
            self.kcur = min(self.K, self.kcur + 1)

            Uc = self.U[:, :self.kcur]
            dc = self.dvec[:self.kcur]
            evec = dc - (self.w @ Uc)

            G = Uc.T @ Uc
            G.flat[::G.shape[0] + 1] += self.cfg.fapa_delta

            # Fast APA-style projection: iterative CG, no matrix inverse and no np.linalg.solve.
            z = solve_projection_cg(G, evec, n_iter=self.cfg.fapa_cg_iters)
            update = Uc @ z

            up_norm = float(np.linalg.norm(update))
            w_norm = float(np.linalg.norm(self.w))
            max_up = self.cfg.fapa_update_clip * (w_norm + 1e-3)
            if up_norm > max_up:
                update *= max_up / (up_norm + 1e-12)

            self.w += mu_eff * update
            echo_block[n] = y
            err_block[n] = e

        diag = block_diagnostic_ffts(self.prev_ref, ref_block, mic_block, err_block, echo_block, self.cfg.diag_fft_len)
        self.prev_ref = ref_block.copy()
        return {"echo_block": echo_block, "err_block": err_block, **diag}


def make_aec(aec_type: str, cfg: HarnessConfig) -> AECBase:
    aec_type = aec_type.lower()
    if aec_type == "pbfdaf":
        return VerifiedPBFDAF(cfg)
    if aec_type == "mdf":
        return MDFEchoCanceller(cfg)
    if aec_type == "nlms":
        return NLMSEchoCanceller(cfg)
    if aec_type == "sbnlms":
        return SubbandNLMSEchoCanceller(cfg)
    if aec_type == "fblms":
        return FBLMSEchoCanceller(cfg)
    if aec_type == "apa":
        return APAEchoCanceller(cfg)
    if aec_type == "fxapa_direct":
        return FilteredXAPADirectEchoCanceller(cfg)
    if aec_type == "fapa":
        return FastAPAEchoCanceller(cfg)
    raise ValueError(aec_type)



class SoftDTD:
    STATE_IDLE = 0
    STATE_DOUBLE_TALK = 1
    STATE_NEAR_END_ONLY = 2
    STATE_FAR_END_ONLY = 3
    STATE_HOLD = 4

    STATE_NAME_TO_CODE = {
        "IDLE": STATE_IDLE,
        "DOUBLE_TALK": STATE_DOUBLE_TALK,
        "NEAR_END_ONLY": STATE_NEAR_END_ONLY,
        "FAR_END_ONLY": STATE_FAR_END_ONLY,
        "HOLD": STATE_HOLD,
    }

    def __init__(self, cfg: HarnessConfig, backend_name: str = "generic"):
        self.cfg = cfg
        self.backend_name = backend_name
        self.K = cfg.diag_fft_len // 2 + 1
        self.phi_ref = np.zeros(self.K, dtype=np.float64)
        self.phi_mic = np.zeros(self.K, dtype=np.float64)
        self.phi_err = np.zeros(self.K, dtype=np.float64)
        self.phi_echo = np.zeros(self.K, dtype=np.float64)
        self.phi_ref_mic = np.zeros(self.K, dtype=np.complex128)
        self.phi_ref_err = np.zeros(self.K, dtype=np.complex128)
        self.p_dt_smooth = 0.0
        self.state = "IDLE"
        self.hold = 0
        self.block_idx = 0
        freqs = np.fft.rfftfreq(cfg.diag_fft_len, d=1.0 / cfg.fs)
        self.band_mask = ((freqs >= 200.0) & (freqs <= 6000.0)).astype(np.float64)
        self.relaxation, self.hold_scale = self._backend_control_params(backend_name)

    def _backend_control_params(self, backend_name: str) -> tuple[float, float]:
        mapping = {
            "pbfdaf": (1.00, 1.00),
            "mdf":    (0.95, 1.00),
            "nlms":   (1.10, 1.10),
            "sbnlms": (1.00, 1.00),
            "fblms":  (0.95, 1.00),
            "apa":    (0.80, 0.80),
            "fxapa_direct": (0.80, 0.80),
            "fapa":   (0.85, 0.85),
        }
        base = mapping.get(backend_name, (1.0, 1.0))
        return base[0] * self.cfg.backend_control_relaxation, base[1] * self.cfg.backend_hold_scale

    def _state_code(self) -> int:
        return self.STATE_NAME_TO_CODE.get(self.state, self.STATE_IDLE)

    def update(self, ref_fft: Array, mic_fft: Array, err_fft: Array, echo_fft: Array) -> Dict[str, float]:
        a = 0.8
        self.block_idx += 1
        self.phi_ref = a * self.phi_ref + (1.0 - a) * (ref_fft * np.conj(ref_fft)).real
        self.phi_mic = a * self.phi_mic + (1.0 - a) * (mic_fft * np.conj(mic_fft)).real
        self.phi_err = a * self.phi_err + (1.0 - a) * (err_fft * np.conj(err_fft)).real
        self.phi_echo = a * self.phi_echo + (1.0 - a) * (echo_fft * np.conj(echo_fft)).real
        self.phi_ref_mic = a * self.phi_ref_mic + (1.0 - a) * (ref_fft * np.conj(mic_fft))
        self.phi_ref_err = a * self.phi_ref_err + (1.0 - a) * (ref_fft * np.conj(err_fft))

        eps = 1e-12
        coh_ref_mic = np.clip((np.abs(self.phi_ref_mic) ** 2) / (self.phi_ref * self.phi_mic + eps), 0.0, 1.0)
        coh_ref_err = np.clip((np.abs(self.phi_ref_err) ** 2) / (self.phi_ref * self.phi_err + eps), 0.0, 1.0)
        mic_over_echo = np.log((self.phi_mic + eps) / (self.phi_echo + eps))
        err_over_mic = np.log((self.phi_err + eps) / (self.phi_mic + eps))

        # Backend-agnostic double-talk / near-end evidence
        near_end_evidence = np.clip((0.60 - coh_ref_mic) / 0.35, 0.0, 1.0)
        uncorrelated_err = np.clip((0.25 - coh_ref_err) / 0.25, 0.0, 1.0)
        mic_echo_mismatch = np.clip((mic_over_echo - 0.30) / 1.50, 0.0, 1.0)
        strong_err = np.clip((err_over_mic + 0.7) / 0.7, 0.0, 1.0)

        p_dt_bin = np.clip(
            0.55 * near_end_evidence +
            0.30 * uncorrelated_err +
            0.10 * mic_echo_mismatch +
            0.05 * strong_err,
            0.0, 1.0,
        )

        echo_only_conf_bin = (
            np.clip((coh_ref_mic - 0.80) / 0.15, 0.0, 1.0) *
            np.clip((coh_ref_err - 0.45) / 0.25, 0.0, 1.0)
        )

        startup_relax = min(self.block_idx / 64.0, 1.0)
        p_dt_bin *= (1.0 - 0.995 * echo_only_conf_bin)
        p_dt_bin *= startup_relax
        p_dt_bin *= (self.phi_ref > self.cfg.ref_power_floor).astype(np.float64)

        w = np.sqrt(self.phi_ref + eps) * self.band_mask
        p_dt = float(np.sum(w * p_dt_bin) / (np.sum(w) + eps))

        alpha = 0.35 if p_dt > self.p_dt_smooth else 0.96
        self.p_dt_smooth = alpha * self.p_dt_smooth + (1.0 - alpha) * p_dt

        near_only_score = float(np.mean(near_end_evidence))
        far_only_score = float(np.mean(echo_only_conf_bin))

        enter = self.cfg.dt_enter
        exit_ = self.cfg.dt_exit
        hold_blocks = max(1, int(round(self.cfg.hold_blocks * self.hold_scale)))

        if self.state in ("IDLE", "FAR_END_ONLY", "NEAR_END_ONLY"):
            if self.p_dt_smooth > enter:
                self.state = "DOUBLE_TALK"
                self.hold = hold_blocks
            elif near_only_score > 0.55 and far_only_score < 0.20:
                self.state = "NEAR_END_ONLY"
            elif far_only_score > 0.65:
                self.state = "FAR_END_ONLY"
            else:
                self.state = "IDLE"
        elif self.state == "DOUBLE_TALK":
            if self.p_dt_smooth < exit_:
                self.state = "HOLD"
                self.hold = hold_blocks
        elif self.state == "HOLD":
            self.hold -= 1
            if self.p_dt_smooth > enter:
                self.state = "DOUBLE_TALK"
                self.hold = hold_blocks
            elif self.hold <= 0:
                self.state = "FAR_END_ONLY" if far_only_score > 0.5 else "IDLE"

        # Backend-specific adaptation controller
        if self.state in ("IDLE", "FAR_END_ONLY"):
            mu_scale = 1.0
        elif self.state == "NEAR_END_ONLY":
            mu_scale = min(1.0, 0.70 * self.relaxation)
        elif self.state == "HOLD":
            mu_scale = min(1.0, 0.85 * self.relaxation)
        else:
            mu_scale = max(0.02, min(1.0, self.relaxation * (1.0 - self.p_dt_smooth) ** 2.0))

        return {
            "p_dt_smooth": self.p_dt_smooth,
            "mu_scale": mu_scale,
            "state_code": float(self._state_code()),
            "near_only_score": near_only_score,
            "far_only_score": far_only_score,
        }


def run_harness(base_dir: str | Path, aec_type: str, scenario_names: List[str], cfg: HarnessConfig, make_plots: bool = True) -> Dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    out_dir = base_dir / f"outputs_{aec_type}"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    sweep_rows = []
    for i, name in enumerate(scenario_names):
        sim = simulate_scenario(name, fs=cfg.fs, dur_s=cfg.duration_s, seed=100 + i)
        aec = make_aec(aec_type, cfg)
        dtd = SoftDTD(cfg, backend_name=aec_type)
        n_blocks = min(len(sim["ref"]), len(sim["mic"])) // cfg.block_len
        n = n_blocks * cfg.block_len
        ref = sim["ref"][:n]
        mic = sim["mic"][:n]
        err = np.zeros(n, dtype=np.float64)
        p_dt_trace = np.zeros(n_blocks, dtype=np.float64)
        mu_trace = np.ones(n_blocks, dtype=np.float64)
        state_trace = np.zeros(n_blocks, dtype=np.float64)
        mu_scale = 1.0
        for b in range(n_blocks):
            s = b * cfg.block_len
            e = s + cfg.block_len
            blk = aec.process_block(ref[s:e], mic[s:e], mu_scale=mu_scale)
            if cfg.bypass_dtd:
                det_mu = 1.0
                det_p = 0.0
            else:
                det = dtd.update(blk["ref_fft"], blk["mic_fft"], blk["err_fft"], blk["echo_fft"])
                det_mu = float(det["mu_scale"])
                det_p = float(det["p_dt_smooth"])
            mu_scale = det_mu
            err[s:e] = blk["err_block"]
            p_dt_trace[b] = det_p
            mu_trace[b] = mu_scale
        if make_plots:
            plot_scenario(plot_dir, name, ref, mic, err, p_dt_trace, mu_trace, state_trace, cfg, sim["dt_truth"][:n])
        row = compute_metrics(name, mic, err, p_dt_trace, sim["dt_truth"][:n], cfg, threshold=0.5)
        row["scenario"] = name
        row["aec_type"] = aec_type
        summary_rows.append(row)
        for th in [0.35, 0.45, 0.50, 0.55, 0.65]:
            m = compute_metrics(name, mic, err, p_dt_trace, sim["dt_truth"][:n], cfg, threshold=th)
            m["scenario"] = name
            m["aec_type"] = aec_type
            sweep_rows.append(m)
    summary = pd.DataFrame(summary_rows)
    sweep = pd.DataFrame(sweep_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)
    return {"summary": summary, "sweep": sweep}


def run_no_dtd_far_end_only(base_dir: str | Path, aec_type: str, cfg: HarnessConfig, make_plots: bool = True) -> pd.DataFrame:
    base_dir = Path(base_dir)
    out_dir = base_dir / f"outputs_{aec_type}_no_dtd"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
    sim = simulate_scenario("far_end_only", fs=cfg.fs, dur_s=cfg.duration_s, seed=100)
    aec = make_aec(aec_type, cfg)
    n_blocks = min(len(sim["ref"]), len(sim["mic"])) // cfg.block_len
    n = n_blocks * cfg.block_len
    ref = sim["ref"][:n]
    mic = sim["mic"][:n]
    err = np.zeros(n, dtype=np.float64)
    p_dt_trace = np.zeros(n_blocks, dtype=np.float64)
    mu_trace = np.ones(n_blocks, dtype=np.float64)
    state_trace = np.full(n_blocks, 3.0, dtype=np.float64)
    for b in range(n_blocks):
        s = b * cfg.block_len
        e = s + cfg.block_len
        blk = aec.process_block(ref[s:e], mic[s:e], mu_scale=1.0)
        err[s:e] = blk["err_block"]
    if make_plots:
        plot_scenario(plot_dir, "far_end_only_no_dtd", ref, mic, err, p_dt_trace, mu_trace, state_trace, cfg, sim["dt_truth"][:n])
    late75 = int(0.75 * len(mic))
    late90 = int(0.90 * len(mic))
    df = pd.DataFrame([{
        "aec_type": aec_type,
        "scenario": "far_end_only_no_dtd",
        "erle_db_full": erle_db(mic, err, 0),
        "erle_db_late_75pct": erle_db(mic, err, late75),
        "erle_db_late_90pct": erle_db(mic, err, late90),
    }])
    df.to_csv(out_dir / "benchmark_summary.csv", index=False)
    return df


def write_rollup(summary: pd.DataFrame, out_dir: Path, cfg: HarnessConfig) -> pd.DataFrame:
    df = summary.copy()
    if not cfg.include_road_noise_in_rollup:
        df = df[df["scenario"] != "road_noise_robustness"].copy()
    rollup = (
        df.groupby("aec_type", as_index=False)
        .agg(
            mean_erle_db_late=("erle_db_late", "mean"),
            mean_convergence_time_ms=("convergence_time_ms", "mean"),
            mean_dt_miss_rate=("dt_miss_rate", "mean"),
            mean_dt_false_alarm_rate=("dt_false_alarm_rate", "mean"),
        )
        .sort_values("mean_erle_db_late", ascending=False)
    )
    rollup.to_csv(out_dir / "rollup.csv", index=False)
    return rollup


def run_comparison(base_dir: str | Path, aec_types: List[str], scenario_names: List[str], cfg: HarnessConfig) -> pd.DataFrame:
    rows = []
    for aec_type in aec_types:
        results = run_harness(base_dir, aec_type, scenario_names, cfg, make_plots=True)
        rows.append(results["summary"])
    leaderboard = pd.concat(rows, ignore_index=True)
    out_dir = Path(base_dir) / "comparison_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    best_rows = []
    for scenario in leaderboard["scenario"].unique():
        sdf = leaderboard[leaderboard["scenario"] == scenario].sort_values("erle_db_late", ascending=False)
        best_rows.append(sdf.iloc[0])
    pd.DataFrame(best_rows).to_csv(out_dir / "best_by_scenario.csv", index=False)
    rollup = (
        leaderboard[leaderboard["scenario"] != "road_noise_robustness"].groupby("aec_type", as_index=False)
        .agg(
            mean_erle_db_late=("erle_db_late", "mean"),
            mean_convergence_time_ms=("convergence_time_ms", "mean"),
            mean_dt_miss_rate=("dt_miss_rate", "mean"),
            mean_dt_false_alarm_rate=("dt_false_alarm_rate", "mean"),
        )
        .sort_values("mean_erle_db_late", ascending=False)
    )
    rollup.to_csv(out_dir / "rollup.csv", index=False)
    return leaderboard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEC harness with swappable backend algorithms")
    p.add_argument("--aec", choices=["pbfdaf", "mdf", "nlms", "sbnlms", "fblms", "apa", "fxapa_direct", "fapa"], default="pbfdaf")
    p.add_argument("--scenario", default="all", help="Scenario name or 'all'")
    p.add_argument("--compare", default="", help="Comma-separated AEC types to compare, e.g. pbfdaf,nlms,apa,mdf")
    p.add_argument("--no-dtd", action="store_true", help="Bypass DTD/adaptation control")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--block-len", type=int, default=128)
    p.add_argument("--erle-avg-blocks", type=int, default=64)
    p.add_argument("--convergence-target-db", type=float, default=20.0)
    p.add_argument("--convergence-hold-blocks", type=int, default=8)
    p.add_argument("--convergence-start-mode", choices=["far_end_onset", "path_change", "manual_block"], default="far_end_onset")
    p.add_argument("--convergence-manual-start-block", type=int, default=0)
    p.add_argument("--include-road-noise-in-rollup", action="store_true")
    p.add_argument("--nlms-filter-len", type=int, default=640)
    p.add_argument("--sbnlms-filter-len", type=int, default=640)
    p.add_argument("--sbnlms-num-taps", type=int, default=5)
    p.add_argument("--fblms-filter-len", type=int, default=640)
    p.add_argument("--apa-filter-len", type=int, default=640)
    p.add_argument("--apa-order", type=int, default=4)
    p.add_argument("--fxapa-filter-len", type=int, default=640)
    p.add_argument("--fxapa-order", type=int, default=4)
    p.add_argument("--fxapa-secondary-delay", type=int, default=0)
    p.add_argument("--fxapa-secondary-gain", type=float, default=1.0)
    p.add_argument("--fapa-filter-len", type=int, default=640)
    p.add_argument("--fapa-order", type=int, default=4)
    p.add_argument("--fapa-cg-iters", type=int, default=8)
    p.add_argument("--fapa-update-clip", type=float, default=1e9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HarnessConfig(
        duration_s=args.duration,
        block_len=args.block_len,
        diag_fft_len=2 * args.block_len,
        erle_avg_blocks=args.erle_avg_blocks,
        bypass_dtd=args.no_dtd,
        nlms_filter_len=args.nlms_filter_len,
        sbnlms_filter_len=args.sbnlms_filter_len,
        sbnlms_num_taps=args.sbnlms_num_taps,
        fblms_filter_len=args.fblms_filter_len,
        apa_filter_len=args.apa_filter_len,
        apa_order=args.apa_order,
        fxapa_filter_len=args.fxapa_filter_len,
        fxapa_order=args.fxapa_order,
        fxapa_secondary_delay=args.fxapa_secondary_delay,
        fxapa_secondary_gain=args.fxapa_secondary_gain,
        fapa_filter_len=args.fapa_filter_len,
        fapa_order=args.fapa_order,
        fapa_cg_iters=args.fapa_cg_iters,
        fapa_update_clip=args.fapa_update_clip,
        convergence_target_db=args.convergence_target_db,
        convergence_hold_blocks=args.convergence_hold_blocks,
        convergence_start_mode=args.convergence_start_mode,
        convergence_manual_start_block=args.convergence_manual_start_block,
        include_road_noise_in_rollup=args.include_road_noise_in_rollup,
    )

    scenarios = [
        "far_end_only",
        "balanced_double_talk",
        "near_end_dominant",
        "path_variability",
        "nonlinear_playback",
        "road_noise_robustness",
        "delay_jump",
        "voices",
    ]
    scenario_names = scenarios if args.scenario == "all" else [args.scenario]
    base_dir = Path(__file__).resolve().parent

    if args.compare:
        aec_types = [x.strip() for x in args.compare.split(",") if x.strip()]
        leaderboard = run_comparison(base_dir, aec_types, scenario_names, cfg)
        out_dir = base_dir / "comparison_outputs"
        print("Wrote:")
        print(out_dir / "leaderboard.csv")
        print(out_dir / "best_by_scenario.csv")
        print(out_dir / "rollup.csv")
        print()
        print("Comparison leaderboard:")
        print(leaderboard.to_string(index=False))
        return

    results = run_harness(base_dir, args.aec, scenario_names, cfg)
    bench = run_no_dtd_far_end_only(base_dir, args.aec, cfg)
    out_dir = base_dir / f"outputs_{args.aec}"
    rollup = write_rollup(results["summary"], out_dir, cfg)
    print("Wrote:")
    print(base_dir / f"outputs_{args.aec}" / "summary.csv")
    print(base_dir / f"outputs_{args.aec}" / "threshold_sweep.csv")
    print(base_dir / f"outputs_{args.aec}" / "plots")
    print(base_dir / f"outputs_{args.aec}" / "rollup.csv")
    print(base_dir / f"outputs_{args.aec}_no_dtd" / "benchmark_summary.csv")
    print(base_dir / f"outputs_{args.aec}_no_dtd" / "plots")
    print()
    print("Main harness summary:")
    print(results["summary"].to_string(index=False))
    print()
    print("Rollup:")
    print(rollup.to_string(index=False))
    print()
    print("No-DTD far-end-only benchmark:")
    print(bench.to_string(index=False))


if __name__ == "__main__":
    main()
