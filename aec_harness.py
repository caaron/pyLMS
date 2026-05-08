
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.io.wavfile import read
from scipy.signal import resample
import urllib.request
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUBLIC_VOICE_URLS = {
    "libri_female_103_1240": "https://homepages.inf.ed.ac.uk/htang2/notes/speech-samples/103-1240-0000.wav",
    "libri_male_1034_121119": "https://homepages.inf.ed.ac.uk/htang2/notes/speech-samples/1034-121119-0000.wav",
    "libri_female_1069_133699": "https://homepages.inf.ed.ac.uk/htang2/notes/speech-samples/1069-133699-0000.wav",
    "libri_male_1081_125237": "https://homepages.inf.ed.ac.uk/htang2/notes/speech-samples/1081-125237-0000.wav",
}

SYNTHETIC_SCENARIOS = [
    "far_end_only",
    "balanced_double_talk",
    "near_end_dominant",
    "path_variability",
    "nonlinear_playback",
    "road_noise_robustness",
    "delay_jump",
]

VOICE_SCENARIOS = [
    "voices",
    "voices_swapped",
    "voices_download_public",
    "voices_libri_0",
    "voices_libri_1",
    "voices_libri_cross",
]

DEFAULT_SCENARIOS = SYNTHETIC_SCENARIOS + [
    "voices",
    "voices_swapped",
    "voices_download_public",
]

ALL_KNOWN_SCENARIOS = SYNTHETIC_SCENARIOS + VOICE_SCENARIOS

def maybe_download_public_voice_files(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in PUBLIC_VOICE_URLS.items():
        out = data_dir / f'{name}.wav'
        if out.exists():
            continue
        try:
            urllib.request.urlretrieve(url, out)
            print(f'Downloaded {out}')
        except Exception as exc:
            print(f'Could not download {url}: {exc}')


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

    dt_enter: float = 0.80
    dt_exit: float = 0.54
    hold_blocks: int = 10
    ref_power_floor: float = 1e-8

    # DTD activity/noise-floor tracking; broadband-aware, not voice-only.
    dtd_ref_active_floor_rel: float = 6.0
    dtd_mic_active_floor_rel: float = 6.0
    dtd_noise_smooth: float = 0.995
    dtd_low_activity_release: float = 0.70
    dtd_echo_veto_strength: float = 0.999
    dtd_startup_blocks: int = 96
    dtd_enter_persist_blocks: int = 4
    dtd_activity_gate_power: float = 2.4

    # Explicit DTD state classifier thresholds
    dtd_ref_activity_score_enter: float = 0.12
    dtd_mic_activity_score_enter: float = 0.12
    dtd_far_coh_enter: float = 0.55
    dtd_far_coh_exit: float = 0.35
    dtd_residual_coh_near: float = 0.25
    dtd_residual_excess_enter: float = 0.35
    dtd_near_excess_enter: float = 0.35
    dtd_dt_enter_persist_blocks: int = 3
    dtd_state_smooth: float = 0.80

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


def read_mono_wav(path: Path, fs: int) -> Array:
    Fs, data = read(str(path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float64) / float(np.iinfo(data.dtype).max + 1)
    else:
        x = data.astype(np.float64)
    if Fs != fs:
        x = resample(x, int(len(x) * fs / Fs))
    return x.astype(np.float64)


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
    elif name.startswith("voices"):
        data_dir = Path(__file__).resolve().parent / "data"
        voice_pairs = {
            "voices": ("Hill_noisy.wav", "armstrong_noisy.wav"),
            "voices_swapped": ("armstrong_noisy.wav", "Hill_noisy.wav"),
        }

        public_files = [
            "libri_female_103_1240.wav",
            "libri_male_1034_121119.wav",
            "libri_female_1069_133699.wav",
            "libri_male_1081_125237.wav",
        ]
        if all((data_dir / f).exists() for f in public_files):
            voice_pairs.update({
                "voices_libri_0": (public_files[0], public_files[1]),
                "voices_libri_1": (public_files[2], public_files[3]),
                "voices_libri_cross": (public_files[0], public_files[3]),
            })

        if name == "voices_download_public":
            maybe_download_public_voice_files(data_dir)
            return simulate_scenario("voices_libri_0", fs=fs, dur_s=None, seed=seed)

        if name not in voice_pairs:
            raise ValueError(f"Unknown voice scenario {name}; available={sorted(voice_pairs)}")

        ref = read_mono_wav(data_dir / voice_pairs[name][0], fs)
        near = read_mono_wav(data_dir / voice_pairs[name][1], fs)

        # Voice scenarios are intentionally processed at full overlap length.
        # This keeps A/B comparisons from silently changing when --duration is set
        # for synthetic scenarios. The effective length is the minimum of the two
        # voice files being paired.
        L = min(len(ref), len(near))
        ref = ref[:L]
        near = near[:L]

        additive_noise = np.zeros(L, dtype=np.float64)
        # Voice scenarios are real overlap cases. Use energy-derived truth instead of all-zero truth.
        # This gives the false-alarm metric a meaningful low-activity baseline.
        ref_blk = np.convolve(ref, np.ones(max(1, fs // 100)) / max(1, fs // 100), mode="same") ** 2
        near_blk = np.convolve(near, np.ones(max(1, fs // 100)) / max(1, fs // 100), mode="same") ** 2
        ref_thr = max(1e-10, 0.01 * np.percentile(ref_blk, 95))
        near_thr = max(1e-10, 0.01 * np.percentile(near_blk, 95))
        dt_truth = ((ref_blk > ref_thr) & (near_blk > near_thr)).astype(np.float64)

        echo = np.convolve(ref, ir, mode="full")[:L]
        mic = echo + near + additive_noise
        return {"ref": ref.astype(np.float64), "mic": mic.astype(np.float64), "dt_truth": dt_truth, "ir": ir}
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



def estimate_ops_per_sample(aec_type: str, cfg: HarnessConfig) -> float:
    """
    Rough real-operation-count estimate per input sample.

    This is for relative comparison in the harness, not cycle-accurate Hexagon profiling.
    It excludes Python overhead, plotting, DTD, memory movement, and cache effects.
    """
    aec_type = aec_type.lower()
    L = cfg.block_len

    if aec_type == "nlms":
        return float(4 * cfg.nlms_filter_len)

    if aec_type == "apa":
        M = cfg.apa_filter_len
        K = cfg.apa_order
        return float((2 * M) + (2 * M * K) + (M * K * K) + (K ** 3) + (2 * M * K))

    if aec_type == "fxapa_direct":
        M = cfg.fxapa_filter_len
        K = cfg.fxapa_order
        return float((2 * M) + (2 * M * K) + (M * K * K) + (K ** 3) + (2 * M * K))

    if aec_type == "fapa":
        M = cfg.fapa_filter_len
        K = cfg.fapa_order
        I = cfg.fapa_cg_iters
        return float((2 * M) + (M * K * K) + (2 * I * K * K) + (2 * M * K))

    if aec_type == "fblms":
        M = cfg.fblms_filter_len
        N = next_pow2(M + cfg.block_len - 1)
        Kfft = N // 2 + 1
        fft_cost = 2.5 * N * np.log2(N)
        block_ops = 3 * fft_cost + 12 * Kfft
        return float(block_ops / L)

    if aec_type == "pbfdaf":
        N = 2 * cfg.block_len
        Kfft = N // 2 + 1
        P = cfg.pbfdaf_n_partitions
        fft_cost = 2.5 * N * np.log2(N)
        block_ops = 3 * fft_cost + P * (2 * fft_cost + 20 * Kfft)
        return float(block_ops / L)

    if aec_type == "mdf":
        N = 2 * cfg.block_len
        Kfft = N // 2 + 1
        P = cfg.mdf_n_partitions
        fft_cost = 2.5 * N * np.log2(N)
        block_ops = 3 * fft_cost + P * (2 * fft_cost + 20 * Kfft)
        return float(block_ops / L)

    if aec_type == "sbnlms":
        N = 2 * cfg.block_len
        Kfft = N // 2 + 1
        P = getattr(cfg, "sbnlms_num_taps", 5)
        fft_cost = 2.5 * N * np.log2(N)
        block_ops = 4 * fft_cost + 24 * Kfft * P
        return float(block_ops / L)

    return float("nan")



def compute_metrics(scenario_name: str, aec_type: str, mic: Array, err: Array, p_dt_smooth: Array, dt_truth: Array, cfg: HarnessConfig, threshold: float = 0.5) -> Dict[str, float]:
    gt = sample_to_block_labels(dt_truth, cfg.block_len)
    pred = (p_dt_smooth >= threshold).astype(np.float64)
    dt_mask = gt > 0
    not_dt_mask = ~dt_mask
    late = int(0.75 * len(mic))
    _, avg_erle = compute_erle_series(mic, err, cfg)
    conv_ms, converged = convergence_time_ms(avg_erle, scenario_name, cfg)
    return {
        "threshold": float(threshold),
        "ops_per_sample_est": estimate_ops_per_sample(aec_type, cfg),
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
        self.ref_floor = np.ones(self.K, dtype=np.float64) * cfg.ref_power_floor
        self.mic_floor = np.ones(self.K, dtype=np.float64) * cfg.ref_power_floor
        self.err_floor = np.ones(self.K, dtype=np.float64) * cfg.ref_power_floor
        self.p_dt_smooth = 0.0
        self.state = "IDLE"
        self.hold = 0
        self.dt_enter_count = 0
        self.block_idx = 0
        freqs = np.fft.rfftfreq(cfg.diag_fft_len, d=1.0 / cfg.fs)
        self.band_mask = ((freqs >= 80.0) & (freqs <= min(0.45 * cfg.fs, 7200.0))).astype(np.float64)
        self.relaxation, self.hold_scale = self._backend_control_params(backend_name)

    def _backend_control_params(self, backend_name: str) -> tuple[float, float]:
        mapping = {
            "pbfdaf": (1.00, 1.00),
            "mdf": (0.95, 1.00),
            "nlms": (1.10, 1.10),
            "sbnlms": (1.00, 1.00),
            "fblms": (0.95, 1.00),
            "apa": (0.80, 0.80),
            "fxapa_direct": (0.80, 0.80),
            "fapa": (0.85, 0.85),
        }
        base = mapping.get(backend_name, (1.0, 1.0))
        return base[0] * self.cfg.backend_control_relaxation, base[1] * self.cfg.backend_hold_scale

    def _state_code(self) -> int:
        return self.STATE_NAME_TO_CODE.get(self.state, self.STATE_IDLE)

    @staticmethod
    def _soft_activity(power: Array, floor: Array, rel: float) -> Array:
        ratio = power / (rel * floor + 1e-18)
        return np.clip((ratio - 1.0) / 3.0, 0.0, 1.0)

    @staticmethod
    def _wmean(x: Array, w: Array) -> float:
        return float(np.sum(w * x) / (np.sum(w) + 1e-12))

    def update(self, ref_fft: Array, mic_fft: Array, err_fft: Array, echo_fft: Array) -> Dict[str, float]:
        a = 0.8
        eps = 1e-12
        self.block_idx += 1

        inst_ref = (ref_fft * np.conj(ref_fft)).real
        inst_mic = (mic_fft * np.conj(mic_fft)).real
        inst_err = (err_fft * np.conj(err_fft)).real
        inst_echo = (echo_fft * np.conj(echo_fft)).real

        self.phi_ref = a * self.phi_ref + (1.0 - a) * inst_ref
        self.phi_mic = a * self.phi_mic + (1.0 - a) * inst_mic
        self.phi_err = a * self.phi_err + (1.0 - a) * inst_err
        self.phi_echo = a * self.phi_echo + (1.0 - a) * inst_echo
        self.phi_ref_mic = a * self.phi_ref_mic + (1.0 - a) * (ref_fft * np.conj(mic_fft))
        self.phi_ref_err = a * self.phi_ref_err + (1.0 - a) * (ref_fft * np.conj(err_fft))

        ns = self.cfg.dtd_noise_smooth
        self.ref_floor = np.where(self.phi_ref < 6.0 * self.ref_floor + eps, ns * self.ref_floor + (1.0 - ns) * np.maximum(self.phi_ref, self.cfg.ref_power_floor), self.ref_floor)
        self.mic_floor = np.where(self.phi_mic < 6.0 * self.mic_floor + eps, ns * self.mic_floor + (1.0 - ns) * np.maximum(self.phi_mic, self.cfg.ref_power_floor), self.mic_floor)
        self.err_floor = np.where(self.phi_err < 6.0 * self.err_floor + eps, ns * self.err_floor + (1.0 - ns) * np.maximum(self.phi_err, self.cfg.ref_power_floor), self.err_floor)

        ref_act_bin = self._soft_activity(self.phi_ref, self.ref_floor, self.cfg.dtd_ref_active_floor_rel)
        mic_act_bin = self._soft_activity(self.phi_mic, self.mic_floor, self.cfg.dtd_mic_active_floor_rel)
        err_act_bin = self._soft_activity(self.phi_err, self.err_floor, self.cfg.dtd_mic_active_floor_rel)

        band = self.band_mask
        ref_w = np.sqrt(self.phi_ref + eps) * band
        mic_w = np.sqrt(self.phi_mic + eps) * band
        err_w = np.sqrt(self.phi_err + eps) * band

        ref_activity = self._wmean(ref_act_bin, ref_w)
        mic_activity = self._wmean(mic_act_bin, mic_w)
        ref_active = ref_activity > self.cfg.dtd_ref_activity_score_enter
        mic_active = mic_activity > self.cfg.dtd_mic_activity_score_enter

        coh_ref_mic = np.clip((np.abs(self.phi_ref_mic) ** 2) / (self.phi_ref * self.phi_mic + eps), 0.0, 1.0)
        coh_ref_err = np.clip((np.abs(self.phi_ref_err) ** 2) / (self.phi_ref * self.phi_err + eps), 0.0, 1.0)

        activity_gate = np.power(np.sqrt(ref_act_bin * mic_act_bin), self.cfg.dtd_activity_gate_power)
        joint_w = np.sqrt(self.phi_ref + eps) * activity_gate * band
        err_joint_w = np.sqrt(self.phi_err + eps) * activity_gate * band
        mic_joint_w = np.sqrt(self.phi_mic + eps) * activity_gate * band

        far_coh = self._wmean(coh_ref_mic, joint_w)
        residual_coh = self._wmean(coh_ref_err, err_joint_w)

        residual_excess_bin = np.clip((self.phi_err - self.phi_echo) / (self.phi_mic + eps), 0.0, 2.0)
        near_excess_bin = np.clip((self.phi_mic - self.phi_echo) / (self.phi_mic + eps), 0.0, 2.0)
        residual_excess = self._wmean(residual_excess_bin, err_joint_w)
        near_excess = self._wmean(near_excess_bin, mic_joint_w)

        p_dt_raw = 0.0
        if ref_active and mic_active:
            incoh_mic = np.clip((self.cfg.dtd_far_coh_enter - far_coh) / max(0.05, self.cfg.dtd_far_coh_enter), 0.0, 1.0)
            incoh_resid = np.clip((self.cfg.dtd_residual_coh_near - residual_coh) / max(0.05, self.cfg.dtd_residual_coh_near), 0.0, 1.0)
            resid_term = np.clip(residual_excess / max(0.05, self.cfg.dtd_residual_excess_enter), 0.0, 1.0)
            near_term = np.clip(near_excess / max(0.05, self.cfg.dtd_near_excess_enter), 0.0, 1.0)
            p_dt_raw = float(np.clip(0.35 * incoh_mic + 0.25 * incoh_resid + 0.25 * resid_term + 0.15 * near_term, 0.0, 1.0))

        p_dt_raw *= min(self.block_idx / max(1.0, float(self.cfg.dtd_startup_blocks)), 1.0)
        self.p_dt_smooth = self.cfg.dtd_state_smooth * self.p_dt_smooth + (1.0 - self.cfg.dtd_state_smooth) * p_dt_raw

        if not ref_active and not mic_active:
            next_state = "IDLE"
            self.dt_enter_count = 0
            self.hold = 0
        elif not ref_active and mic_active:
            next_state = "NEAR_END_ONLY"
            self.dt_enter_count = 0
            self.hold = 0
        elif ref_active and not mic_active:
            next_state = "IDLE"
            self.dt_enter_count = 0
            self.hold = 0
        else:
            far_like = far_coh >= self.cfg.dtd_far_coh_enter and residual_excess < self.cfg.dtd_residual_excess_enter
            dt_like = (
                self.p_dt_smooth >= self.cfg.dt_enter or
                (
                    far_coh < self.cfg.dtd_far_coh_exit and
                    residual_coh < self.cfg.dtd_residual_coh_near and
                    (residual_excess > self.cfg.dtd_residual_excess_enter or near_excess > self.cfg.dtd_near_excess_enter)
                )
            )
            if dt_like:
                self.dt_enter_count += 1
                if self.dt_enter_count >= self.cfg.dtd_dt_enter_persist_blocks:
                    next_state = "DOUBLE_TALK"
                    self.hold = max(1, int(round(self.cfg.hold_blocks * self.hold_scale)))
                else:
                    next_state = self.state
            else:
                self.dt_enter_count = 0
                if self.state == "DOUBLE_TALK" and self.p_dt_smooth >= self.cfg.dt_exit:
                    next_state = "DOUBLE_TALK"
                elif self.state == "DOUBLE_TALK":
                    next_state = "HOLD"
                    self.hold = max(1, int(round(self.cfg.hold_blocks * self.hold_scale)))
                elif self.state == "HOLD" and self.hold > 0:
                    self.hold -= 1
                    next_state = "HOLD"
                else:
                    next_state = "FAR_END_ONLY" if far_like or far_coh > self.cfg.dtd_far_coh_exit else "IDLE"

        self.state = next_state

        if self.state in ("IDLE", "FAR_END_ONLY"):
            mu_scale = 1.0
        elif self.state == "NEAR_END_ONLY":
            mu_scale = min(1.0, 0.70 * self.relaxation)
        elif self.state == "HOLD":
            mu_scale = min(1.0, 0.85 * self.relaxation)
        else:
            mu_scale = max(0.02, min(1.0, self.relaxation * (1.0 - self.p_dt_smooth) ** 2.0))

        return {
            "p_dt_smooth": float(self.p_dt_smooth),
            "mu_scale": float(mu_scale),
            "state_code": float(self._state_code()),
            "near_only_score": float(near_excess),
            "far_only_score": float(far_coh),
        }


class LogMelNeuralDTD:
    """
    Skeleton DTD inspired by a neural SPP/VAD front end plus classic AEC guards.

    Drop-in intent:
      - Same output dictionary as SoftDTD.update(...)
      - Provides process(...) as the primary API
      - Provides update(...) as an alias so the existing harness call site can be
        switched by changing only the constructed class.

    This is intentionally a skeleton: the neural model hook defaults to a
    deterministic heuristic SPP estimator, but can be replaced by ONNX Runtime,
    PyTorch, TFLite, etc. later.

    Expected inputs are the existing diagnostic FFTs returned by the AEC blocks:
      ref_fft  : far-end/reference spectrum
      mic_fft  : microphone spectrum
      err_fft  : residual/error spectrum after echo estimate subtraction
      echo_fft : estimated echo spectrum

    The class combines:
      - log-mel near-end speech-presence evidence from mic/residual spectra
      - far-end activity from reference power
      - echo-likeness / coherence between ref and mic
      - residual coherence between ref and residual error
      - residual/near-end excess over estimated echo
      - optional harmonicity/pitch-confidence proxy
      - attack/release smoothing, hysteresis, persistence, and hold
    """

    STATE_IDLE = SoftDTD.STATE_IDLE
    STATE_DOUBLE_TALK = SoftDTD.STATE_DOUBLE_TALK
    STATE_NEAR_END_ONLY = SoftDTD.STATE_NEAR_END_ONLY
    STATE_FAR_END_ONLY = SoftDTD.STATE_FAR_END_ONLY
    STATE_HOLD = SoftDTD.STATE_HOLD
    STATE_NAME_TO_CODE = SoftDTD.STATE_NAME_TO_CODE

    def __init__(self, cfg: HarnessConfig, backend_name: str = "generic", spp_model=None):
        self.cfg = cfg
        self.backend_name = backend_name
        self.spp_model = spp_model  # Optional callable: features -> dict or float.

        self.K = cfg.diag_fft_len // 2 + 1
        self.freqs = np.fft.rfftfreq(cfg.diag_fft_len, d=1.0 / cfg.fs)
        self.band_mask = ((self.freqs >= 80.0) & (self.freqs <= min(0.45 * cfg.fs, 7200.0))).astype(np.float64)

        # Mel feature config. For cfg.diag_fft_len == 2 * block_len, this is a
        # short-window approximation rather than a textbook 25 ms / 10 ms VAD
        # frontend. It is still useful for A/B inside this harness. In production,
        # run a dedicated 25 ms window / 10 ms hop frontend and pass SPP here.
        self.n_mels = 48
        self.mel_fb = self._make_mel_filterbank(self.n_mels, cfg.diag_fft_len, cfg.fs)
        self.logmel_mean = np.zeros(self.n_mels, dtype=np.float64)
        self.logmel_var = np.ones(self.n_mels, dtype=np.float64)
        self.logmel_norm_smooth = 0.995

        # Smoothed spectra and cross spectra.
        self.phi_ref = np.zeros(self.K, dtype=np.float64)
        self.phi_mic = np.zeros(self.K, dtype=np.float64)
        self.phi_err = np.zeros(self.K, dtype=np.float64)
        self.phi_echo = np.zeros(self.K, dtype=np.float64)
        self.phi_ref_mic = np.zeros(self.K, dtype=np.complex128)
        self.phi_ref_err = np.zeros(self.K, dtype=np.complex128)

        self.ref_floor = np.ones(self.K, dtype=np.float64) * cfg.ref_power_floor
        self.mic_floor = np.ones(self.K, dtype=np.float64) * cfg.ref_power_floor
        self.err_floor = np.ones(self.K, dtype=np.float64) * cfg.ref_power_floor

        self.p_near_smooth = 0.0
        self.p_bin_spp = np.zeros(self.K, dtype=np.float64)
        self.p_dt_smooth = 0.0
        self.state = "IDLE"
        self.hold = 0
        self.dt_enter_count = 0
        self.block_idx = 0

        self.relaxation, self.hold_scale = SoftDTD(cfg, backend_name)._backend_control_params(backend_name)

    @staticmethod
    def _hz_to_mel(f_hz: Array | float) -> Array | float:
        return 2595.0 * np.log10(1.0 + np.asarray(f_hz) / 700.0)

    @staticmethod
    def _mel_to_hz(mel: Array | float) -> Array | float:
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    @classmethod
    def _make_mel_filterbank(cls, n_mels: int, n_fft: int, fs: int, fmin: float = 80.0, fmax: float | None = None) -> Array:
        fmax = float(fmax if fmax is not None else min(0.45 * fs, 7600.0))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        mel_pts = np.linspace(cls._hz_to_mel(fmin), cls._hz_to_mel(fmax), n_mels + 2)
        hz_pts = cls._mel_to_hz(mel_pts)
        fb = np.zeros((n_mels, len(freqs)), dtype=np.float64)
        for m in range(n_mels):
            left, center, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            up = (freqs - left) / max(center - left, 1e-12)
            down = (right - freqs) / max(right - center, 1e-12)
            fb[m] = np.maximum(0.0, np.minimum(up, down))
        fb /= np.maximum(np.sum(fb, axis=1, keepdims=True), 1e-12)
        return fb

    @staticmethod
    def _soft_activity(power: Array, floor: Array, rel: float) -> Array:
        ratio = power / (rel * floor + 1e-18)
        return np.clip((ratio - 1.0) / 3.0, 0.0, 1.0)

    @staticmethod
    def _wmean(x: Array, w: Array) -> float:
        return float(np.sum(w * x) / (np.sum(w) + 1e-12))

    def _state_code(self) -> int:
        return self.STATE_NAME_TO_CODE.get(self.state, self.STATE_IDLE)

    def _logmel(self, power: Array) -> Array:
        mel_e = self.mel_fb @ np.maximum(power, 0.0)
        logmel = np.log(np.maximum(mel_e, 1e-12))

        # Online normalization for model stability. In production this should use
        # training-set mean/std or a carefully designed streaming CMVN.
        a = self.logmel_norm_smooth
        delta = logmel - self.logmel_mean
        self.logmel_mean = a * self.logmel_mean + (1.0 - a) * logmel
        self.logmel_var = a * self.logmel_var + (1.0 - a) * (delta ** 2)
        return (logmel - self.logmel_mean) / np.sqrt(self.logmel_var + 1e-6)

    def _harmonicity_proxy(self, power: Array) -> float:
        """Cheap tonal/harmonic confidence proxy, not a real pitch tracker."""
        band = (self.freqs >= 120.0) & (self.freqs <= 3500.0)
        p = np.maximum(power[band], 1e-18)
        if p.size < 4:
            return 0.0
        peakiness = np.max(p) / (np.mean(p) + 1e-18)
        flatness = np.exp(np.mean(np.log(p))) / (np.mean(p) + 1e-18)
        return float(np.clip(0.15 * np.log1p(peakiness) + 0.85 * (1.0 - flatness), 0.0, 1.0))

    def _heuristic_spp_model(self, features: Dict[str, Array | float]) -> Dict[str, Array | float]:
        """
        Stand-in for a trained causal TCN/GRU/Conformer SPP model.

        Replace this with e.g.:
            out = onnx_session.run(None, {"logmel": features["logmel"][None, None, :]})
            return {"frame_spp": out_frame, "bin_spp": out_bins}
        """
        logmel = np.asarray(features["logmel"], dtype=np.float64)
        harmonicity = float(features["harmonicity"])
        residual_excess = float(features["residual_excess"])
        residual_incoh = float(features["residual_incoh"])

        # Speech-like energy + residual unexplained-by-echo evidence. This is not
        # meant to be SOTA; it lets the skeleton run before a neural model exists.
        energy_score = float(np.clip((np.mean(logmel) + 1.0) / 3.0, 0.0, 1.0))
        frame_spp = np.clip(
            0.30 * energy_score +
            0.25 * harmonicity +
            0.25 * residual_excess +
            0.20 * residual_incoh,
            0.0,
            1.0,
        )

        # Per-bin SPP proxy: speech likely where residual exceeds echo/floor and
        # lies in the voice band. A real model should produce this directly.
        err_power = np.asarray(features["err_power"], dtype=np.float64)
        err_floor = np.asarray(features["err_floor"], dtype=np.float64)
        bin_spp = self._soft_activity(err_power, err_floor, self.cfg.dtd_mic_active_floor_rel) * self.band_mask
        return {"frame_spp": float(frame_spp), "bin_spp": np.clip(bin_spp, 0.0, 1.0)}

    def process(self, ref_fft: Array, mic_fft: Array, err_fft: Array, echo_fft: Array) -> Dict[str, float]:
        a = 0.8
        eps = 1e-12
        self.block_idx += 1

        inst_ref = (ref_fft * np.conj(ref_fft)).real
        inst_mic = (mic_fft * np.conj(mic_fft)).real
        inst_err = (err_fft * np.conj(err_fft)).real
        inst_echo = (echo_fft * np.conj(echo_fft)).real

        self.phi_ref = a * self.phi_ref + (1.0 - a) * inst_ref
        self.phi_mic = a * self.phi_mic + (1.0 - a) * inst_mic
        self.phi_err = a * self.phi_err + (1.0 - a) * inst_err
        self.phi_echo = a * self.phi_echo + (1.0 - a) * inst_echo
        self.phi_ref_mic = a * self.phi_ref_mic + (1.0 - a) * (ref_fft * np.conj(mic_fft))
        self.phi_ref_err = a * self.phi_ref_err + (1.0 - a) * (ref_fft * np.conj(err_fft))

        ns = self.cfg.dtd_noise_smooth
        self.ref_floor = np.where(self.phi_ref < 6.0 * self.ref_floor + eps, ns * self.ref_floor + (1.0 - ns) * np.maximum(self.phi_ref, self.cfg.ref_power_floor), self.ref_floor)
        self.mic_floor = np.where(self.phi_mic < 6.0 * self.mic_floor + eps, ns * self.mic_floor + (1.0 - ns) * np.maximum(self.phi_mic, self.cfg.ref_power_floor), self.mic_floor)
        self.err_floor = np.where(self.phi_err < 6.0 * self.err_floor + eps, ns * self.err_floor + (1.0 - ns) * np.maximum(self.phi_err, self.cfg.ref_power_floor), self.err_floor)

        ref_act_bin = self._soft_activity(self.phi_ref, self.ref_floor, self.cfg.dtd_ref_active_floor_rel)
        mic_act_bin = self._soft_activity(self.phi_mic, self.mic_floor, self.cfg.dtd_mic_active_floor_rel)
        err_act_bin = self._soft_activity(self.phi_err, self.err_floor, self.cfg.dtd_mic_active_floor_rel)

        band = self.band_mask
        ref_activity = self._wmean(ref_act_bin, np.sqrt(self.phi_ref + eps) * band)
        mic_activity = self._wmean(mic_act_bin, np.sqrt(self.phi_mic + eps) * band)
        ref_active = ref_activity > self.cfg.dtd_ref_activity_score_enter
        mic_active = mic_activity > self.cfg.dtd_mic_activity_score_enter

        coh_ref_mic = np.clip((np.abs(self.phi_ref_mic) ** 2) / (self.phi_ref * self.phi_mic + eps), 0.0, 1.0)
        coh_ref_err = np.clip((np.abs(self.phi_ref_err) ** 2) / (self.phi_ref * self.phi_err + eps), 0.0, 1.0)

        activity_gate = np.power(np.sqrt(ref_act_bin * mic_act_bin), self.cfg.dtd_activity_gate_power)
        joint_w = np.sqrt(self.phi_ref + eps) * activity_gate * band
        err_joint_w = np.sqrt(self.phi_err + eps) * activity_gate * band
        mic_joint_w = np.sqrt(self.phi_mic + eps) * activity_gate * band

        far_coh = self._wmean(coh_ref_mic, joint_w)
        residual_coh = self._wmean(coh_ref_err, err_joint_w)
        residual_incoh = float(np.clip(1.0 - residual_coh / max(0.05, self.cfg.dtd_residual_coh_near), 0.0, 1.0))

        residual_excess_bin = np.clip((self.phi_err - self.phi_echo) / (self.phi_mic + eps), 0.0, 2.0)
        near_excess_bin = np.clip((self.phi_mic - self.phi_echo) / (self.phi_mic + eps), 0.0, 2.0)
        residual_excess = self._wmean(residual_excess_bin, err_joint_w)
        near_excess = self._wmean(near_excess_bin, mic_joint_w)

        logmel_mic = self._logmel(self.phi_mic)
        harmonicity = self._harmonicity_proxy(self.phi_err)
        features = {
            "logmel": logmel_mic,
            "harmonicity": harmonicity,
            "ref_activity": float(ref_activity),
            "mic_activity": float(mic_activity),
            "far_coh": float(far_coh),
            "residual_coh": float(residual_coh),
            "residual_incoh": residual_incoh,
            "residual_excess": float(np.clip(residual_excess / max(0.05, self.cfg.dtd_residual_excess_enter), 0.0, 1.0)),
            "near_excess": float(np.clip(near_excess / max(0.05, self.cfg.dtd_near_excess_enter), 0.0, 1.0)),
            "err_power": self.phi_err,
            "err_floor": self.err_floor,
        }

        model_out = self.spp_model(features) if self.spp_model is not None else self._heuristic_spp_model(features)
        if isinstance(model_out, dict):
            p_near_raw = float(model_out.get("frame_spp", 0.0))
            self.p_bin_spp = np.asarray(model_out.get("bin_spp", self.p_bin_spp), dtype=np.float64)
        else:
            p_near_raw = float(model_out)

        p_near_raw = float(np.clip(p_near_raw, 0.0, 1.0))

        # Fast attack, slower release in block-time. With 128 samples at 16 kHz,
        # one block is 8 ms, so these approximate ~20 ms attack / ~200 ms release.
        if p_near_raw > self.p_near_smooth:
            alpha_near = np.exp(-self.cfg.block_len / max(1.0, 0.020 * self.cfg.fs))
        else:
            alpha_near = np.exp(-self.cfg.block_len / max(1.0, 0.200 * self.cfg.fs))
        self.p_near_smooth = alpha_near * self.p_near_smooth + (1.0 - alpha_near) * p_near_raw

        # Double-talk requires far-end activity AND near-end SPP. Echo-likeness
        # suppresses false double-talk on clean far-end-only echo.
        far_not_echo_only = np.clip((self.cfg.dtd_far_coh_enter - far_coh) / max(0.05, self.cfg.dtd_far_coh_enter), 0.0, 1.0)
        excess_evidence = np.clip(0.65 * features["residual_excess"] + 0.35 * features["near_excess"], 0.0, 1.0)
        p_dt_raw = 0.0
        if ref_active:
            p_dt_raw = float(np.clip(self.p_near_smooth * (0.45 + 0.35 * far_not_echo_only + 0.20 * excess_evidence), 0.0, 1.0))

        p_dt_raw *= min(self.block_idx / max(1.0, float(self.cfg.dtd_startup_blocks)), 1.0)
        self.p_dt_smooth = self.cfg.dtd_state_smooth * self.p_dt_smooth + (1.0 - self.cfg.dtd_state_smooth) * p_dt_raw

        if not ref_active and not mic_active:
            next_state = "IDLE"
            self.dt_enter_count = 0
            self.hold = 0
        elif not ref_active and self.p_near_smooth > self.cfg.dt_exit:
            next_state = "NEAR_END_ONLY"
            self.dt_enter_count = 0
            self.hold = 0
        elif ref_active and self.p_dt_smooth >= self.cfg.dt_enter:
            self.dt_enter_count += 1
            if self.dt_enter_count >= self.cfg.dtd_dt_enter_persist_blocks:
                next_state = "DOUBLE_TALK"
                self.hold = max(1, int(round(self.cfg.hold_blocks * self.hold_scale)))
            else:
                next_state = self.state
        elif self.state == "DOUBLE_TALK" and self.p_dt_smooth >= self.cfg.dt_exit:
            next_state = "DOUBLE_TALK"
        elif self.state == "DOUBLE_TALK":
            next_state = "HOLD"
            self.hold = max(1, int(round(self.cfg.hold_blocks * self.hold_scale)))
        elif self.state == "HOLD" and self.hold > 0:
            self.hold -= 1
            next_state = "HOLD"
        elif ref_active:
            next_state = "FAR_END_ONLY" if far_coh > self.cfg.dtd_far_coh_exit else "IDLE"
            self.dt_enter_count = 0
        else:
            next_state = "IDLE"
            self.dt_enter_count = 0

        self.state = next_state

        if self.state in ("IDLE", "FAR_END_ONLY"):
            mu_scale = 1.0
        elif self.state == "NEAR_END_ONLY":
            mu_scale = min(1.0, 0.70 * self.relaxation)
        elif self.state == "HOLD":
            mu_scale = min(1.0, 0.85 * self.relaxation)
        else:
            mu_scale = max(0.02, min(1.0, self.relaxation * (1.0 - self.p_dt_smooth) ** 2.0))

        return {
            "p_dt_smooth": float(self.p_dt_smooth),
            "mu_scale": float(mu_scale),
            "state_code": float(self._state_code()),
            "near_only_score": float(self.p_near_smooth),
            "far_only_score": float(far_coh),
        }

    def update(self, ref_fft: Array, mic_fft: Array, err_fft: Array, echo_fft: Array) -> Dict[str, float]:
        """Compatibility shim for the current harness call site."""
        return self.process(ref_fft, mic_fft, err_fft, echo_fft)

def make_dtd(dtd_type: str, cfg: HarnessConfig, backend_name: str = "generic"):
    dtd_type = dtd_type.lower()
    if dtd_type == "soft":
        return SoftDTD(cfg, backend_name=backend_name)
    if dtd_type in ("logmel", "logmel_neural", "neural_logmel"):
        return LogMelNeuralDTD(cfg, backend_name=backend_name)
    raise ValueError(f"Unknown DTD type {dtd_type!r}; expected soft or logmel")


def run_harness(base_dir: str | Path, aec_type: str, scenario_names: List[str], cfg: HarnessConfig, dtd_type: str = "soft", make_plots: bool = True) -> Dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    dtd_type = dtd_type.lower()
    out_dir = base_dir / f"outputs_{aec_type}_{dtd_type}"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    do_plots = True
    if make_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    sweep_rows = []
    for i, name in enumerate(scenario_names):
        sim = simulate_scenario(name, fs=cfg.fs, dur_s=cfg.duration_s, seed=100 + i)
        aec = make_aec(aec_type, cfg)
        dtd = make_dtd(dtd_type, cfg, backend_name=aec_type)
        n_blocks = min(len(sim["ref"]), len(sim["mic"])) // cfg.block_len
        n = n_blocks * cfg.block_len
        ref = sim["ref"][:n]
        mic = sim["mic"][:n]
        if do_plots:
            tmpfig, ax = plt.subplots(1)
            ax.plot(ref,label="ref")
            ax.plot(mic, label="mic")
            ax.plot(sim["dt_truth"],label="DT")
            ax.legend()
            plt.title(f"{aec_type}_{dtd_type}_{name}")
            plt.pause(.1)
            plt.pause(.1)

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
                det_state = 3.0
            else:
                det = dtd.update(blk["ref_fft"], blk["mic_fft"], blk["err_fft"], blk["echo_fft"])
                det_mu = float(det["mu_scale"])
                det_p = float(det["p_dt_smooth"])
                det_state = float(det["state_code"])
            mu_scale = det_mu
            err[s:e] = blk["err_block"]
            p_dt_trace[b] = det_p
            mu_trace[b] = mu_scale
            state_trace[b] = det_state
        if make_plots:
            plot_scenario(plot_dir, name, ref, mic, err, p_dt_trace, mu_trace, state_trace, cfg, sim["dt_truth"][:n])
        row = compute_metrics(name, aec_type, mic, err, p_dt_trace, sim["dt_truth"][:n], cfg, threshold=0.5)
        row["scenario"] = name
        row["aec_type"] = aec_type
        row["dtd_type"] = dtd_type
        summary_rows.append(row)
        for th in [0.35, 0.45, 0.50, 0.55, 0.65]:
            m = compute_metrics(name, aec_type, mic, err, p_dt_trace, sim["dt_truth"][:n], cfg, threshold=th)
            m["scenario"] = name
            m["aec_type"] = aec_type
            m["dtd_type"] = dtd_type
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
    group_cols = ["aec_type"]
    if "dtd_type" in df.columns:
        group_cols.append("dtd_type")
    rollup = (
        df.groupby(group_cols, as_index=False)
        .agg(
            mean_erle_db_late=("erle_db_late", "mean"),
            mean_convergence_time_ms=("convergence_time_ms", "mean"),
            ops_per_sample_est=("ops_per_sample_est", "mean"),
            mean_dt_miss_rate=("dt_miss_rate", "mean"),
            mean_dt_false_alarm_rate=("dt_false_alarm_rate", "mean"),
        )
        .sort_values("mean_erle_db_late", ascending=False)
    )
    rollup.to_csv(out_dir / "rollup.csv", index=False)
    return rollup


def run_comparison(base_dir: str | Path, aec_types: List[str], dtd_types: List[str], scenario_names: List[str], cfg: HarnessConfig) -> pd.DataFrame:
    rows = []
    for aec_type in aec_types:
        for dtd_type in dtd_types:
            results = run_harness(base_dir, aec_type, scenario_names, cfg, dtd_type=dtd_type, make_plots=True)
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
        leaderboard[leaderboard["scenario"] != "road_noise_robustness"].groupby(["aec_type", "dtd_type"], as_index=False)
        .agg(
            mean_erle_db_late=("erle_db_late", "mean"),
            mean_convergence_time_ms=("convergence_time_ms", "mean"),
            ops_per_sample_est=("ops_per_sample_est", "mean"),
            mean_dt_miss_rate=("dt_miss_rate", "mean"),
            mean_dt_false_alarm_rate=("dt_false_alarm_rate", "mean"),
        )
        .sort_values("mean_erle_db_late", ascending=False)
    )
    rollup.to_csv(out_dir / "rollup.csv", index=False)
    return leaderboard


def parse_csv_choices(value: str, valid: List[str], *, default_all: List[str], label: str) -> List[str]:
    """Parse comma-separated CLI values with support for 'all'."""
    parts = [x.strip() for x in str(value).split(",") if x.strip()]
    if not parts or parts == ["all"]:
        return list(default_all)
    if "all" in parts:
        # Allow mixes like --scenario all,voices_libri_0 by expanding all and de-duping.
        expanded = list(default_all) + [x for x in parts if x != "all"]
        parts = expanded
    bad = sorted(set(parts) - set(valid))
    if bad:
        raise ValueError(f"Unknown {label}(s): {bad}; expected 'all' or one or more of {sorted(valid)}")
    out: List[str] = []
    seen = set()
    for item in parts:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEC harness with swappable backend algorithms")
    p.add_argument("--aec", choices=["pbfdaf", "mdf", "nlms", "sbnlms", "fblms", "apa", "fxapa_direct", "fapa"], default="pbfdaf")
    p.add_argument("--dtd", choices=["soft", "logmel"], default="soft", help="DTD implementation to use")
    p.add_argument("--scenario", default="all", help="Scenario name, comma-separated scenario list, or 'all'")
    p.add_argument("--compare", default="", help="Comma-separated AEC types to compare, e.g. pbfdaf,nlms,apa,mdf")
    p.add_argument("--compare-dtd", default="", help="Comma-separated DTD types to compare, e.g. soft,logmel")
    p.add_argument("--no-dtd", action="store_true", help="Bypass DTD/adaptation control")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--block-len", type=int, default=128)
    p.add_argument("--erle-avg-blocks", type=int, default=64)
    p.add_argument("--convergence-target-db", type=float, default=20.0)
    p.add_argument("--convergence-hold-blocks", type=int, default=8)
    p.add_argument("--convergence-start-mode", choices=["far_end_onset", "path_change", "manual_block"], default="far_end_onset")
    p.add_argument("--convergence-manual-start-block", type=int, default=0)
    p.add_argument("--include-road-noise-in-rollup", action="store_true")
    p.add_argument("--dtd-ref-active-floor-rel", type=float, default=6.0)
    p.add_argument("--dtd-mic-active-floor-rel", type=float, default=6.0)
    p.add_argument("--dtd-low-activity-release", type=float, default=0.70)
    p.add_argument("--dtd-echo-veto-strength", type=float, default=0.999)
    p.add_argument("--dtd-startup-blocks", type=int, default=96)
    p.add_argument("--dtd-enter-persist-blocks", type=int, default=4)
    p.add_argument("--dtd-activity-gate-power", type=float, default=2.4)
    p.add_argument("--dtd-ref-activity-score-enter", type=float, default=0.12)
    p.add_argument("--dtd-mic-activity-score-enter", type=float, default=0.12)
    p.add_argument("--dtd-far-coh-enter", type=float, default=0.55)
    p.add_argument("--dtd-far-coh-exit", type=float, default=0.35)
    p.add_argument("--dtd-residual-coh-near", type=float, default=0.25)
    p.add_argument("--dtd-residual-excess-enter", type=float, default=0.35)
    p.add_argument("--dtd-near-excess-enter", type=float, default=0.35)
    p.add_argument("--dtd-dt-enter-persist-blocks", type=int, default=3)
    p.add_argument("--dtd-state-smooth", type=float, default=0.80)
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
        dtd_ref_active_floor_rel=args.dtd_ref_active_floor_rel,
        dtd_mic_active_floor_rel=args.dtd_mic_active_floor_rel,
        dtd_low_activity_release=args.dtd_low_activity_release,
        dtd_echo_veto_strength=args.dtd_echo_veto_strength,
        dtd_startup_blocks=args.dtd_startup_blocks,
        dtd_enter_persist_blocks=args.dtd_enter_persist_blocks,
        dtd_activity_gate_power=args.dtd_activity_gate_power,
        dtd_ref_activity_score_enter=args.dtd_ref_activity_score_enter,
        dtd_mic_activity_score_enter=args.dtd_mic_activity_score_enter,
        dtd_far_coh_enter=args.dtd_far_coh_enter,
        dtd_far_coh_exit=args.dtd_far_coh_exit,
        dtd_residual_coh_near=args.dtd_residual_coh_near,
        dtd_residual_excess_enter=args.dtd_residual_excess_enter,
        dtd_near_excess_enter=args.dtd_near_excess_enter,
        dtd_dt_enter_persist_blocks=args.dtd_dt_enter_persist_blocks,
        dtd_state_smooth=args.dtd_state_smooth,
    )

    scenario_names = parse_csv_choices(
        args.scenario,
        ALL_KNOWN_SCENARIOS,
        default_all=DEFAULT_SCENARIOS,
        label="scenario",
    )
    base_dir = Path(__file__).resolve().parent

    if args.compare or args.compare_dtd:
        aec_types = [x.strip() for x in args.compare.split(",") if x.strip()] if args.compare else [args.aec]
        dtd_types = [x.strip() for x in args.compare_dtd.split(",") if x.strip()] if args.compare_dtd else [args.dtd]
        valid_dtd = {"soft", "logmel"}
        bad_dtd = sorted(set(dtd_types) - valid_dtd)
        if bad_dtd:
            raise ValueError(f"Unknown DTD type(s): {bad_dtd}; expected one or more of {sorted(valid_dtd)}")
        leaderboard = run_comparison(base_dir, aec_types, dtd_types, scenario_names, cfg)
        out_dir = base_dir / "comparison_outputs"
        print("Wrote:")
        print(out_dir / "leaderboard.csv")
        print(out_dir / "best_by_scenario.csv")
        print(out_dir / "rollup.csv")
        print()
        print("Comparison leaderboard:")
        print(leaderboard.to_string(index=False))
        return

    results = run_harness(base_dir, args.aec, scenario_names, cfg, dtd_type=args.dtd)
    bench = run_no_dtd_far_end_only(base_dir, args.aec, cfg)
    out_dir = base_dir / f"outputs_{args.aec}_{args.dtd}"
    rollup = write_rollup(results["summary"], out_dir, cfg)
    print("Wrote:")
    print(base_dir / f"outputs_{args.aec}_{args.dtd}" / "summary.csv")
    print(base_dir / f"outputs_{args.aec}_{args.dtd}" / "threshold_sweep.csv")
    print(base_dir / f"outputs_{args.aec}_{args.dtd}" / "plots")
    print(base_dir / f"outputs_{args.aec}_{args.dtd}" / "rollup.csv")
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
